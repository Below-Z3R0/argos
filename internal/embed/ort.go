package embed

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/Below-Z3R0/argos/internal/registry"
	"github.com/Below-Z3R0/argos/internal/resources"
)

// ORTOptions configures a model session.
type ORTOptions struct {
	ModelPath     string
	TokenizerPath string
	LibraryPath   string // libonnxruntime.so; empty uses the system path
	Name          string
	Dims          int
	MaxTokens     int
	Pooling       string // last_token | mean | cls
	QueryInstruct string
	Prefixes      registry.Prefixes
	Threads       int
	BatchSize     int    // passages per forward pass; 1 disables batching
	Device        string // auto | cpu | gpu
	GPUIndex      int    // -1 = let auto choose
}

// ORT runs a transformer encoder/decoder exported to ONNX.
//
// The export used by Qwen3-Embedding keeps the KV cache in the graph, so the
// session expects 2*N empty past_key_values tensors and returns N present
// ones. They are created here and never used for pooling.
type ORT struct {
	session   *ort.DynamicAdvancedSession
	tok       *tokenizers.Tokenizer
	opts      ORTOptions
	inputs    []string
	outputs   []string
	kvInputs  []string
	kvShape   ort.Shape
	kvTensors map[int][]ort.Value
	modelDims int
	dims      int
	indexLast int
	provider  string
	mu        sync.Mutex
}

var (
	ortOnce sync.Once
	ortErr  error
)

func initORT(library string) error {
	ortOnce.Do(func() {
		if library == "" {
			library = os.Getenv("ARGOS_ORT_LIB")
		}
		if library == "" {
			library = discoverRuntime()
		}
		ort.SetSharedLibraryPath(library)
		ortErr = ort.InitializeEnvironment()
	})
	return ortErr
}

// discoverRuntime finds the ONNX Runtime to use. The system package may be
// older than the API the Go binding needs, so the bundled copy wins.
func discoverRuntime() string {
	return DiscoverRuntime(os.Getenv("ARGOS_HOME"))
}

// DiscoverRuntime resolves the runtime library: explicit env first, then the
// copy bundled under <home>/runtime, then the system one.
func DiscoverRuntime(home string) string {
	if lib := os.Getenv("ARGOS_ORT_LIB"); lib != "" {
		return lib
	}
	if home != "" {
		matches, _ := filepath.Glob(filepath.Join(home, "runtime", "onnxruntime-*", "lib", "libonnxruntime.so"))
		if len(matches) > 0 {
			sort.Strings(matches)
			return matches[len(matches)-1]
		}
	}
	return "/usr/lib/libonnxruntime.so"
}

// NewORT opens the model and prepares the session.
func NewORT(opts ORTOptions) (*ORT, error) {
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = 512
	}
	if opts.Pooling == "" {
		opts.Pooling = "last_token"
	}
	if err := initORT(opts.LibraryPath); err != nil {
		return nil, fmt.Errorf("onnxruntime: %w", err)
	}

	tok, err := tokenizers.FromFile(opts.TokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("embed: load tokenizer %s: %w", opts.TokenizerPath, err)
	}

	infos, outs, err := ort.GetInputOutputInfo(opts.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("embed: inspect model: %w", err)
	}
	var inputNames, outputNames []string
	var kvInputs []string
	var kvHeads, kvDim int
	for _, in := range infos {
		inputNames = append(inputNames, in.Name)
		if strings.HasPrefix(in.Name, "past_key_values.") {
			kvInputs = append(kvInputs, in.Name)
			if len(in.Dimensions) == 4 && in.Dimensions[1] > 0 && in.Dimensions[3] > 0 {
				kvHeads = int(in.Dimensions[1])
				kvDim = int(in.Dimensions[3])
			}
		}
	}
	// Only last_hidden_state is requested. The export also returns 56
	// present.* tensors for the KV cache; asking for them forces the runtime
	// to materialize ~180 MB per 800-token sequence of data nobody reads.
	lastIndex := -1
	modelDims := 0
	for _, out := range outs {
		if out.Name == "last_hidden_state" {
			outputNames = append(outputNames, out.Name)
			lastIndex = 0
			if len(out.Dimensions) > 0 {
				modelDims = int(out.Dimensions[len(out.Dimensions)-1])
			}
		}
	}
	if lastIndex < 0 {
		return nil, errors.New("embed: model has no last_hidden_state output")
	}
	if opts.Dims == 0 {
		opts.Dims = modelDims
	}
	if modelDims == 0 {
		return nil, errors.New("embed: cannot determine the model dimension")
	}
	if opts.Dims > modelDims {
		return nil, fmt.Errorf("embed: dims %d exceeds the model's %d", opts.Dims, modelDims)
	}

	so, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer so.Destroy()
	if opts.Threads > 0 {
		if err := so.SetIntraOpNumThreads(opts.Threads); err != nil {
			return nil, err
		}
	}
	if err := so.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		return nil, err
	}
	provider, err := attachProvider(so, opts)
	if err != nil {
		return nil, err
	}

	sess, err := ort.NewDynamicAdvancedSession(opts.ModelPath, inputNames, outputNames, so)
	if err != nil {
		return nil, fmt.Errorf("embed: open session: %w", err)
	}

	if opts.BatchSize <= 0 {
		opts.BatchSize = 1
	}
	o := &ORT{
		session:   sess,
		tok:       tok,
		opts:      opts,
		inputs:    inputNames,
		outputs:   outputNames,
		kvInputs:  kvInputs,
		kvShape:   ort.NewShape(1, int64(kvHeads), 0, int64(kvDim)),
		kvTensors: map[int][]ort.Value{},
		modelDims: modelDims,
		dims:      opts.Dims,
		indexLast: lastIndex,
		provider:  provider,
	}
	if _, err := o.kvFor(1); err != nil {
		sess.Destroy()
		return nil, err
	}
	return o, nil
}

// Close releases the session and the tokenizer.
func (o *ORT) Close() error {
	for _, batch := range o.kvTensors {
		for _, kv := range batch {
			_ = kv.Destroy()
		}
	}
	_ = o.tok.Close()
	return o.session.Destroy()
}

// Provider reports the execution provider actually in use.
func (o *ORT) Provider() string { return o.provider }

// attachProvider appends the execution provider the device policy asks for.
// `auto` falls back to CPU when the runtime has no build for the GPU; `gpu`
// fails instead of silently running on CPU.
func attachProvider(so *ort.SessionOptions, opts ORTOptions) (string, error) {
	if opts.Device == "cpu" || opts.Device == "" {
		return "CPU", nil
	}
	gpus := resources.Detect()
	if len(gpus) == 0 {
		if opts.Device == "gpu" {
			return "", errors.New("embed: gpu requested but no GPU was detected")
		}
		return "CPU", nil
	}
	var gpu resources.GPU
	if opts.GPUIndex >= 0 && opts.GPUIndex < len(gpus) {
		gpu = gpus[opts.GPUIndex]
	} else if best, ok := resources.Best(gpus); ok {
		gpu = best
	}
	ep := gpu.Provider()
	if ep == "CPU" {
		return "CPU", nil
	}
	var err error
	switch ep {
	case "CUDA":
		err = so.AppendExecutionProvider("CUDAExecutionProvider", map[string]string{
			"device_id": strconv.Itoa(gpu.Index),
		})
	case "ROCM":
		err = so.AppendExecutionProvider("ROCMExecutionProvider", nil)
	case "OpenVINO":
		err = so.AppendExecutionProvider("OpenVINOExecutionProvider", map[string]string{
			"device_type": "GPU",
		})
	}
	if err == nil {
		return ep, nil
	}
	if opts.Device == "gpu" {
		return "", fmt.Errorf("embed: %s provider unavailable: %w", ep, err)
	}
	return "CPU", nil
}

// Dim implements Embedder.
func (o *ORT) Dim() int { return o.dims }

// Name implements Embedder.
func (o *ORT) Name() string { return o.opts.Name }

// MaxTokens implements Embedder.
func (o *ORT) MaxTokens() int { return o.opts.MaxTokens }

// EmbedQuery embeds a query with the family's instruction or prefix.
func (o *ORT) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	prefixed := o.opts.Prefixes.Query + text
	if o.opts.QueryInstruct != "" {
		prefixed = "Instruct: " + o.opts.QueryInstruct + "\nQuery: " + text
	}
	return o.embedOne(prefixed)
}

// kvFor returns one empty KV tensor per layer, cached by batch size.
func (o *ORT) kvFor(batch int) ([]ort.Value, error) {
	if v, ok := o.kvTensors[batch]; ok {
		return v, nil
	}
	shape := ort.NewShape(int64(batch), o.kvShape[1], 0, o.kvShape[3])
	out := make([]ort.Value, 0, len(o.kvInputs))
	for range o.kvInputs {
		kv, err := ort.NewEmptyTensor[float32](shape)
		if err != nil {
			for _, t := range out {
				t.Destroy()
			}
			return nil, err
		}
		out = append(out, kv)
	}
	o.kvTensors[batch] = out
	return out, nil
}

// EmbedPassages embeds documents. With BatchSize > 1 it pads a batch to the
// longest row and runs one forward pass, which amortizes runtime overhead but
// pays a padding tax; the spike decides whether it wins on this hardware.
func (o *ORT) EmbedPassages(ctx context.Context, texts []string) ([][]float32, error) {
	if o.opts.BatchSize <= 1 {
		return o.embedSequential(ctx, texts)
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += o.opts.BatchSize {
		end := start + o.opts.BatchSize
		if end > len(texts) {
			end = len(texts)
		}
		vecs, err := o.embedBatch(ctx, texts[start:end])
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (o *ORT) embedSequential(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for _, t := range texts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		v, err := o.embedOne(o.opts.Prefixes.Passage + t)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// embedBatch runs one padded batch and returns one vector per input.
func (o *ORT) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	encoded := make([][]int64, len(texts))
	maxLen := 0
	for i, text := range texts {
		enc, err := o.tok.EncodeWithOptionsErr(o.opts.Prefixes.Passage+text, true)
		if err != nil {
			return nil, fmt.Errorf("embed: tokenize: %w", err)
		}
		ids := enc.IDs
		if len(ids) > o.opts.MaxTokens {
			ids = ids[:o.opts.MaxTokens]
		}
		if len(ids) == 0 {
			return nil, errors.New("embed: empty input")
		}
		row := make([]int64, len(ids))
		for j, id := range ids {
			row[j] = int64(id)
		}
		encoded[i] = row
		if len(row) > maxLen {
			maxLen = len(row)
		}
	}

	batch := len(texts)
	inputIDs := make([]int64, batch*maxLen)
	attention := make([]int64, batch*maxLen)
	positions := make([]int64, batch*maxLen)
	for i, row := range encoded {
		for j := 0; j < maxLen; j++ {
			idx := i*maxLen + j
			if j < len(row) {
				inputIDs[idx] = row[j]
				attention[idx] = 1
				positions[idx] = int64(j)
			}
		}
	}

	idsTensor, err := ort.NewTensor(ort.NewShape(int64(batch), int64(maxLen)), inputIDs)
	if err != nil {
		return nil, err
	}
	defer idsTensor.Destroy()
	maskTensor, err := ort.NewTensor(ort.NewShape(int64(batch), int64(maxLen)), attention)
	if err != nil {
		return nil, err
	}
	defer maskTensor.Destroy()
	posTensor, err := ort.NewTensor(ort.NewShape(int64(batch), int64(maxLen)), positions)
	if err != nil {
		return nil, err
	}
	defer posTensor.Destroy()

	tensors := map[string]ort.Value{"input_ids": idsTensor, "attention_mask": maskTensor, "position_ids": posTensor}
	kvs, err := o.kvFor(batch)
	if err != nil {
		return nil, err
	}
	for i, name := range o.kvInputs {
		tensors[name] = kvs[i]
	}
	inValues := make([]ort.Value, len(o.inputs))
	for i, name := range o.inputs {
		inValues[i] = tensors[name]
	}
	outValues := make([]ort.Value, len(o.outputs))
	if err := o.session.Run(inValues, outValues); err != nil {
		return nil, fmt.Errorf("embed: run batch: %w", err)
	}
	hidden, ok := outValues[o.indexLast].(*ort.Tensor[float32])
	if !ok {
		return nil, errors.New("embed: last_hidden_state has an unexpected type")
	}
	defer hidden.Destroy()
	data := hidden.GetData()
	if len(data) < batch*maxLen*o.modelDims {
		return nil, fmt.Errorf("embed: batch output has %d values", len(data))
	}

	out := make([][]float32, batch)
	for i := range encoded {
		n := len(encoded[i]) // right padding: the last real token is at n-1
		row := data[i*maxLen*o.modelDims : i*maxLen*o.modelDims+n*o.modelDims]
		out[i] = o.pool(row, n)
	}
	return out, nil
}

func (o *ORT) embedOne(text string) ([]float32, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	enc, err := o.tok.EncodeWithOptionsErr(text, true)
	if err != nil {
		return nil, fmt.Errorf("embed: tokenize: %w", err)
	}
	ids := enc.IDs
	if len(ids) == 0 {
		return nil, errors.New("embed: empty input")
	}
	if len(ids) > o.opts.MaxTokens {
		ids = ids[:o.opts.MaxTokens]
	}
	n := len(ids)

	inputIDs := make([]int64, n)
	attention := make([]int64, n)
	positions := make([]int64, n)
	for i, id := range ids {
		inputIDs[i] = int64(id)
		attention[i] = 1
		positions[i] = int64(i)
	}

	tensors := map[string]ort.Value{}
	idsTensor, err := ort.NewTensor(ort.NewShape(1, int64(n)), inputIDs)
	if err != nil {
		return nil, err
	}
	defer idsTensor.Destroy()
	maskTensor, err := ort.NewTensor(ort.NewShape(1, int64(n)), attention)
	if err != nil {
		return nil, err
	}
	defer maskTensor.Destroy()
	posTensor, err := ort.NewTensor(ort.NewShape(1, int64(n)), positions)
	if err != nil {
		return nil, err
	}
	defer posTensor.Destroy()

	tensors["input_ids"] = idsTensor
	tensors["attention_mask"] = maskTensor
	tensors["position_ids"] = posTensor

	// Empty KV cache: the export declares it as an input, so it must be fed.
	// The tensors are reused across calls and freed in Close.
	kvs, err := o.kvFor(1)
	if err != nil {
		return nil, err
	}
	for i, name := range o.kvInputs {
		tensors[name] = kvs[i]
	}

	inValues := make([]ort.Value, len(o.inputs))
	for i, name := range o.inputs {
		v, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("embed: no tensor for input %q", name)
		}
		inValues[i] = v
	}
	outValues := make([]ort.Value, len(o.outputs))
	if err := o.session.Run(inValues, outValues); err != nil {
		return nil, fmt.Errorf("embed: run: %w", err)
	}
	for i, v := range outValues {
		if v != nil && i != o.indexLast {
			defer v.Destroy()
		}
	}
	hidden, ok := outValues[o.indexLast].(*ort.Tensor[float32])
	if !ok {
		return nil, errors.New("embed: last_hidden_state has an unexpected type")
	}
	defer hidden.Destroy()

	data := hidden.GetData()
	if len(data) < n*o.modelDims {
		return nil, fmt.Errorf("embed: output has %d values, want %d", len(data), n*o.modelDims)
	}
	return o.pool(data, n), nil
}

// pool reduces the token matrix to one vector and normalizes it.
func (o *ORT) pool(data []float32, n int) []float32 {
	vec := make([]float32, o.modelDims)
	switch o.opts.Pooling {
	case "mean":
		for i := 0; i < n; i++ {
			for d := 0; d < o.modelDims; d++ {
				vec[d] += data[i*o.modelDims+d]
			}
		}
		for d := range vec {
			vec[d] /= float32(n)
		}
	case "cls":
		copy(vec, data[:o.modelDims])
	default: // last_token
		copy(vec, data[(n-1)*o.modelDims:n*o.modelDims])
	}
	// Matryoshka: Qwen3 keeps ~99 % of retrieval quality at half the dims.
	// Truncating after pooling and renormalizing is the documented use.
	if o.dims > 0 && o.dims < len(vec) {
		vec = vec[:o.dims]
	}
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range vec {
			vec[i] = float32(float64(vec[i]) / norm)
		}
	}
	return vec
}
