package runner

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func fakeBin(t *testing.T, script string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "fake-argosctl")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestRunCapturesOutput(t *testing.T) {
	bin := fakeBin(t, `echo "args: $@"; echo "warn" >&2; exit 0`)
	res, err := New(bin, 5*time.Second, 1024).Run(context.Background(), "search", "hola")
	if err != nil {
		t.Fatal(err)
	}
	if res.ExitCode != 0 {
		t.Fatalf("exit %d", res.ExitCode)
	}
	if !strings.Contains(res.Stdout, "search hola") {
		t.Fatalf("stdout %q", res.Stdout)
	}
	if !strings.Contains(res.Stderr, "warn") {
		t.Fatalf("stderr %q", res.Stderr)
	}
}

func TestRunReportsExitCode(t *testing.T) {
	bin := fakeBin(t, `echo "boom"; exit 3`)
	res, err := New(bin, 5*time.Second, 1024).Run(context.Background(), "doctor")
	if err != nil {
		t.Fatal(err)
	}
	if res.ExitCode != 3 {
		t.Fatalf("exit %d", res.ExitCode)
	}
}

func TestRunTimesOut(t *testing.T) {
	bin := fakeBin(t, `sleep 2`)
	res, err := New(bin, 100*time.Millisecond, 1024).Run(context.Background(), "index")
	if err != nil {
		t.Fatal(err)
	}
	if !res.TimedOut {
		t.Fatalf("expected timeout, got %+v", res)
	}
}

func TestOutputIsCapped(t *testing.T) {
	bin := fakeBin(t, `yes x | head -c 5000`)
	res, _ := New(bin, 5*time.Second, 100).Run(context.Background())
	if len(res.Stdout) > 200 {
		t.Fatalf("output not capped: %d bytes", len(res.Stdout))
	}
	if !strings.Contains(res.Stdout, "truncada") {
		t.Fatal("missing truncation marker")
	}
}

func TestMissingBinary(t *testing.T) {
	if _, err := New("", 0, 0).Run(context.Background()); err == nil {
		t.Fatal("expected error")
	}
}
