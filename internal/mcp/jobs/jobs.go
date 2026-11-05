// Package jobs runs long commands in the background so an MCP client can start
// an index and poll instead of blocking a tool call for hours.
package jobs

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/Below-Z3R0/argos/internal/mcp/runner"
)

// Job is one background invocation.
type Job struct {
	ID       string    `json:"id"`
	Command  string    `json:"command"`
	Started  time.Time `json:"started"`
	Finished time.Time `json:"finished"`
	Done     bool      `json:"done"`
	Canceled bool      `json:"canceled,omitempty"`
	ExitCode int       `json:"exit_code"`
	Output   string    `json:"output"`
	Error    string    `json:"error,omitempty"`
}

// Manager keeps the live jobs.
type Manager struct {
	mu      sync.Mutex
	jobs    map[string]*Job
	order   []string
	cancels map[string]context.CancelFunc
	max     int
	runner  *runner.Runner
}

// New returns a manager that keeps the last max jobs.
func New(r *runner.Runner, max int) *Manager {
	if max <= 0 {
		max = 20
	}
	return &Manager{jobs: map[string]*Job{}, cancels: map[string]context.CancelFunc{}, max: max, runner: r}
}

// Start launches a command and returns its job id immediately.
func (m *Manager) Start(ctx context.Context, args ...string) (*Job, error) {
	id, err := randomID()
	if err != nil {
		return nil, err
	}
	job := &Job{ID: id, Command: join(args), Started: time.Now().UTC()}
	runCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	m.mu.Lock()
	m.jobs[id] = job
	m.order = append(m.order, id)
	m.cancels[id] = cancel
	evicted := m.evictLocked()
	m.mu.Unlock()
	_ = evicted

	go func() {
		res, err := m.runner.Run(runCtx, args...)
		m.mu.Lock()
		defer m.mu.Unlock()
		delete(m.cancels, id)
		cancel()
		job.Done = true
		job.Finished = time.Now().UTC()
		job.ExitCode = res.ExitCode
		job.Output = res.Stdout
		if res.Stderr != "" {
			if job.Output != "" {
				job.Output += "\n"
			}
			job.Output += res.Stderr
		}
		if err != nil {
			job.Canceled = errors.Is(err, context.Canceled)
			job.Error = err.Error()
		}
		// Evict again on completion: at start time the old jobs may still be
		// running, and the limit must hold once they finish.
		m.evictLocked()
	}()
	return job, nil
}

// Cancel stops a running job. It returns false when the id is unknown or the
// job already finished.
func (m *Manager) Cancel(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	job, ok := m.jobs[id]
	if !ok || job.Done {
		return false
	}
	cancel, ok := m.cancels[id]
	if !ok {
		return false
	}
	cancel()
	return true
}

// Get returns a snapshot of a job.
func (m *Manager) Get(id string) (Job, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	job, ok := m.jobs[id]
	if !ok {
		return Job{}, false
	}
	return *job, true
}

// List returns every live job, newest first.
func (m *Manager) List() []Job {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Job, 0, len(m.order))
	for i := len(m.order) - 1; i >= 0; i-- {
		if job, ok := m.jobs[m.order[i]]; ok {
			out = append(out, *job)
		}
	}
	return out
}

// evictLocked drops the oldest finished jobs past the limit.
func (m *Manager) evictLocked() []string {
	var removed []string
	for len(m.order) > m.max {
		id := m.order[0]
		job, ok := m.jobs[id]
		if ok && !job.Done {
			break // never drop a running job
		}
		delete(m.jobs, id)
		m.order = m.order[1:]
		removed = append(removed, id)
	}
	return removed
}

func randomID() (string, error) {
	buf := make([]byte, 4)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return hex.EncodeToString(buf), nil
}

func join(args []string) string {
	out := "argosctl"
	for _, a := range args {
		out += " " + a
	}
	return out
}

var _ = fmt.Sprint
