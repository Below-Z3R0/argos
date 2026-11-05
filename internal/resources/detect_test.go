package resources

import (
	"os"
	"path/filepath"
	"testing"
)

func fixture(t *testing.T, cards map[string]map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for name, files := range cards {
		dir := filepath.Join(root, name, "device")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		for file, content := range files {
			if err := os.WriteFile(filepath.Join(dir, file), []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	return root
}

func TestDetectAndProvider(t *testing.T) {
	root := fixture(t, map[string]map[string]string{
		"card0":      {"vendor": "0x8086\n", "uevent": "DRIVER=i915\n"},
		"card1":      {"vendor": "0x10de\n", "uevent": "DRIVER=nvidia\n", "product_name": "RTX 3060\n"},
		"card1-DP-1": {"vendor": "0x10de\n"}, // connector, must be ignored
	})
	gpus := detectIn(root)
	if len(gpus) != 2 {
		t.Fatalf("gpus %+v", gpus)
	}
	if gpus[0].Vendor != "intel" || gpus[0].Provider() != "OpenVINO" {
		t.Fatalf("intel: %+v", gpus[0])
	}
	if gpus[1].Vendor != "nvidia" || gpus[1].Provider() != "CUDA" || !gpus[1].Dedicated() {
		t.Fatalf("nvidia: %+v", gpus[1])
	}
	best, ok := Best(gpus)
	if !ok || best.Vendor != "nvidia" {
		t.Fatalf("best %+v", best)
	}
}

func TestBestFallsBackToIntegrated(t *testing.T) {
	root := fixture(t, map[string]map[string]string{
		"card0": {"vendor": "0x8086\n", "uevent": "DRIVER=i915\n"},
	})
	best, ok := Best(detectIn(root))
	if !ok || best.Provider() != "OpenVINO" {
		t.Fatalf("best %+v ok=%v", best, ok)
	}
}

func TestNoGPU(t *testing.T) {
	if _, ok := Best(detectIn(t.TempDir())); ok {
		t.Fatal("expected no GPU")
	}
}
