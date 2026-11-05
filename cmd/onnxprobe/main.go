// Command onnxprobe prints the inputs and outputs of an ONNX model.
// It is a development helper for the embedding spike, not part of the product.
package main

import (
	"fmt"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "uso: onnxprobe <modelo.onnx>")
		os.Exit(2)
	}
	lib := os.Getenv("ARGOS_ORT_LIB")
	if lib == "" {
		lib = "/usr/lib/libonnxruntime.so"
	}
	ort.SetSharedLibraryPath(lib)
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Fprintln(os.Stderr, "init:", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	inputs, outputs, err := ort.GetInputOutputInfo(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "info:", err)
		os.Exit(1)
	}
	fmt.Println("ENTRADAS")
	for _, in := range inputs {
		fmt.Printf("  %-16s %v %v\n", in.Name, in.Dimensions, in.DataType)
	}
	fmt.Println("SALIDAS")
	for _, out := range outputs {
		fmt.Printf("  %-16s %v %v\n", out.Name, out.Dimensions, out.DataType)
	}
}
