// Package runner executes the argosctl binary.
//
// The MCP server does not reimplement the CLI: it calls it. That keeps one
// source of truth for validation, config and output formats.
package runner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"
)

// Result is the outcome of one invocation.
type Result struct {
	Stdout   string
	Stderr   string
	ExitCode int
	Duration time.Duration
	TimedOut bool
}

// Runner executes a binary with a timeout and an output cap.
type Runner struct {
	Bin       string
	Timeout   time.Duration // 0 = no timeout
	MaxOutput int           // bytes per stream
}

// New returns a runner.
func New(bin string, timeout time.Duration, maxOutput int) *Runner {
	if maxOutput <= 0 {
		maxOutput = 256 << 10
	}
	return &Runner{Bin: bin, Timeout: timeout, MaxOutput: maxOutput}
}

// Run executes argosctl with the given arguments.
func (r *Runner) Run(ctx context.Context, args ...string) (Result, error) {
	if r.Bin == "" {
		return Result{}, errors.New("runner: argosctl binary is not configured")
	}
	runCtx := ctx
	var cancel context.CancelFunc
	if r.Timeout > 0 {
		runCtx, cancel = context.WithTimeout(ctx, r.Timeout)
		defer cancel()
	}

	cmd := exec.CommandContext(runCtx, r.Bin, args...)
	stdout := &limitedBuffer{limit: r.MaxOutput}
	stderr := &limitedBuffer{limit: r.MaxOutput}
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	start := time.Now()
	err := cmd.Run()
	res := Result{
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		Duration: time.Since(start),
	}
	var exitErr *exec.ExitError
	switch {
	case err == nil:
		res.ExitCode = 0
	case errors.As(err, &exitErr):
		res.ExitCode = exitErr.ExitCode()
	default:
		return res, fmt.Errorf("runner: exec %s: %w", r.Bin, err)
	}
	if runCtx.Err() == context.DeadlineExceeded {
		res.TimedOut = true
	}
	return res, nil
}

// limitedBuffer keeps at most limit bytes and records the truncation.
type limitedBuffer struct {
	buf       bytes.Buffer
	limit     int
	truncated bool
}

func (b *limitedBuffer) Write(p []byte) (int, error) {
	if b.truncated {
		return len(p), nil
	}
	remaining := b.limit - b.buf.Len()
	if remaining <= 0 {
		b.truncated = true
		return len(p), nil
	}
	if len(p) > remaining {
		b.buf.Write(p[:remaining])
		b.truncated = true
		return len(p), nil
	}
	return b.buf.Write(p)
}

func (b *limitedBuffer) String() string {
	out := b.buf.String()
	if b.truncated {
		out += "\n… (salida truncada)"
	}
	return strings.TrimSpace(out)
}

// Helper to satisfy io.Writer in tests.
var _ io.Writer = (*limitedBuffer)(nil)
