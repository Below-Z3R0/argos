package jobs

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Below-Z3R0/argos/internal/mcp/runner"
)

func fakeRunner(t *testing.T) *runner.Runner {
	t.Helper()
	dir := t.TempDir()
	bin := filepath.Join(dir, "fake")
	script := "#!/bin/sh\nsleep 0.2\necho \"hecho: $@\"\n"
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return runner.New(bin, 10*time.Second, 1<<20)
}

func TestStartAndPoll(t *testing.T) {
	m := New(fakeRunner(t), 5)
	job, err := m.Start(context.Background(), "index", "--db", "knowledge")
	if err != nil {
		t.Fatal(err)
	}
	if job.ID == "" || job.Done {
		t.Fatalf("job %+v", job)
	}

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		got, ok := m.Get(job.ID)
		if !ok {
			t.Fatal("job disappeared")
		}
		if got.Done {
			if !strings.Contains(got.Output, "index --db knowledge") {
				t.Fatalf("output %q", got.Output)
			}
			if got.ExitCode != 0 {
				t.Fatalf("exit %d", got.ExitCode)
			}
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("job never finished")
}

func TestListAndEviction(t *testing.T) {
	m := New(fakeRunner(t), 2)
	for i := 0; i < 3; i++ {
		if _, err := m.Start(context.Background(), "doctor"); err != nil {
			t.Fatal(err)
		}
	}
	time.Sleep(600 * time.Millisecond)
	list := m.List()
	if len(list) > 2 {
		t.Fatalf("eviction failed: %d jobs", len(list))
	}
	if len(list) == 0 || !list[0].Done {
		t.Fatalf("list %+v", list)
	}
}

func TestCancelRunningJob(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "slow")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nsleep 5\necho done\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	m := New(runner.New(bin, 10*time.Second, 1<<20), 2)
	job, err := m.Start(context.Background(), "index")
	if err != nil {
		t.Fatal(err)
	}
	if !m.Cancel(job.ID) {
		t.Fatal("Cancel returned false for a running job")
	}
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		got, ok := m.Get(job.ID)
		if ok && got.Done {
			if !got.Canceled {
				t.Fatalf("job finished without the canceled flag: %+v", got)
			}
			if m.Cancel(job.ID) {
				t.Fatal("Cancel returned true for a finished job")
			}
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("canceled job never finished")
}

func TestUnknownJob(t *testing.T) {
	m := New(fakeRunner(t), 2)
	if _, ok := m.Get("nope"); ok {
		t.Fatal("unexpected job")
	}
}
