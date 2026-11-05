package embed

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// AvailableProviders reports which execution providers this runtime build can
// actually load. Appending a provider to a SessionOptions is what makes ONNX
// Runtime dlopen its library, so this is a real check and not a guess from the
// model name or the GPU vendor.
func AvailableProviders(libraryPath string) ([]string, error) {
	if err := initORT(libraryPath); err != nil {
		return nil, err
	}
	candidates := []struct {
		name     string
		provider string
		options  map[string]string
	}{
		{"CUDA", "CUDAExecutionProvider", nil},
		{"ROCM", "ROCMExecutionProvider", nil},
		{"OpenVINO", "OpenVINOExecutionProvider", map[string]string{"device_type": "GPU"}},
	}
	var out []string
	for _, c := range candidates {
		so, err := ort.NewSessionOptions()
		if err != nil {
			return out, fmt.Errorf("probe %s: %w", c.name, err)
		}
		err = so.AppendExecutionProvider(c.provider, c.options)
		so.Destroy()
		if err == nil {
			out = append(out, c.name)
		}
	}
	return out, nil
}
