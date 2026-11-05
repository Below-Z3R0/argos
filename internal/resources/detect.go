// Package resources detects the hardware Argos can use.
//
// The point is the execution provider: an NVIDIA card wants CUDA, an AMD card
// ROCm, an Intel GPU OpenVINO, and everything else CPU. Detection is explicit
// so `auto` never silently picks a device that does not exist.
package resources

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// GPU is one detected graphics device.
type GPU struct {
	Index  int
	Vendor string // nvidia | amd | intel | unknown
	Name   string
	Driver string
}

// Provider maps a vendor to the ONNX Runtime execution provider.
func (g GPU) Provider() string {
	switch g.Vendor {
	case "nvidia":
		return "CUDA"
	case "amd":
		return "ROCM"
	case "intel":
		return "OpenVINO"
	default:
		return "CPU"
	}
}

// Dedicated reports whether the card is a discrete GPU.
func (g GPU) Dedicated() bool { return g.Vendor == "nvidia" || g.Vendor == "amd" }

// Detect scans /sys/class/drm for cards.
func Detect() []GPU { return detectIn("/sys/class/drm") }

func detectIn(root string) []GPU {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	var out []GPU
	for _, e := range entries {
		name := e.Name()
		if !strings.HasPrefix(name, "card") || strings.Contains(name, "-") {
			continue
		}
		idx, err := strconv.Atoi(strings.TrimPrefix(name, "card"))
		if err != nil {
			continue
		}
		device := filepath.Join(root, name, "device")
		vendor := vendorOf(readTrim(filepath.Join(device, "vendor")))
		if vendor == "unknown" {
			continue
		}
		out = append(out, GPU{
			Index:  idx,
			Vendor: vendor,
			Name:   readTrim(filepath.Join(device, "product_name")),
			Driver: driverOf(readTrim(filepath.Join(device, "uevent"))),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Index < out[j].Index })
	return out
}

func vendorOf(id string) string {
	switch strings.ToLower(strings.TrimSpace(id)) {
	case "0x10de":
		return "nvidia"
	case "0x1002":
		return "amd"
	case "0x8086":
		return "intel"
	default:
		return "unknown"
	}
}

func driverOf(uevent string) string {
	for _, line := range strings.Split(uevent, "\n") {
		if strings.HasPrefix(line, "DRIVER=") {
			return strings.TrimPrefix(line, "DRIVER=")
		}
	}
	return ""
}

func readTrim(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// Best picks the device for `auto`: a discrete GPU first, then the integrated
// one, then nothing (CPU).
func Best(gpus []GPU) (GPU, bool) {
	for _, g := range gpus {
		if g.Dedicated() {
			return g, true
		}
	}
	if len(gpus) > 0 {
		return gpus[0], true
	}
	return GPU{}, false
}
