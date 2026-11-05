// Package rerank scores (query, passage) pairs with a cross-encoder.
//
// The bge-reranker export used here is an XLM-R encoder: it expects
// <s> query </s> </s> passage </s> and returns one logit per pair. The score is
// squashed with a sigmoid so the output is comparable and filterable.
package rerank

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/Below-Z3R0/argos/internal/resources"
)

// Options configures the model session.
type Options struct {
	ModelPath     string
	TokenizerPath string
	LibraryPath   string
	Device        string // auto | cpu | gpu
	GPUIndex      int
	MaxTokens     int
	Threads       int
	BatchSize     int    // pairs per forward pass
	PairTokens    int    // max tokens per pair; 0 uses MaxTokens
	Format        string // ""/xlmr or qwen3 (ChatML seq-cls)
	Instruction   string // reranker task instruction for qwen3
}

// Reranker is a cross-encoder session.
type Reranker struct {
	session   *ort.DynamicAdvancedSession
	tok       *tokenizers.Tokenizer
	cls, sep  int64
	maxTokens int
	pairLimit int
	batch     int
	provider  string
	format    string
	instruct  string
	mu        sync.Mutex
}

// New opens the model and reads the pair template from the tokenizer.
func New(opts Options) (*Reranker, error) {
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = 1024
	}
	if opts.PairTokens <= 0 {
		opts.PairTokens = 384
	}
	if opts.BatchSize <= 0 {
		opts.BatchSize = 8
	}
	ids := [2]int64{}
	if opts.Format != "qwen3" {
		var err error
		ids, err = specialIDs(opts.TokenizerPath)
		if err != nil {
			return nil, err
		}
	}
	tok, err := tokenizers.FromFile(opts.TokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("rerank: tokenizer: %w", err)
	}

	lib := opts.LibraryPath
	if lib == "" {
		lib = os.Getenv("ARGOS_ORT_LIB")
	}
	if lib == "" {
		lib = "/usr/lib/libonnxruntime.so"
	}
	ort.SetSharedLibraryPath(lib)
	if err := ort.InitializeEnvironment(); err != nil && !strings.Contains(strings.ToLower(err.Error()), "already") {
		return nil, fmt.Errorf("rerank: onnxruntime: %w", err)
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
	provider := attachProvider(so, opts)

	session, err := ort.NewDynamicAdvancedSession(
		opts.ModelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"},
		so,
	)
	if err != nil {
		return nil, fmt.Errorf("rerank: open session: %w", err)
	}
	return &Reranker{
		session: session, tok: tok, cls: ids[0], sep: ids[1],
		maxTokens: opts.MaxTokens, pairLimit: opts.PairTokens, batch: opts.BatchSize,
		provider: provider, format: opts.Format, instruct: opts.Instruction,
	}, nil
}

// Close releases the session.
func (r *Reranker) Close() error {
	_ = r.tok.Close()
	return r.session.Destroy()
}

// Provider reports the execution provider in use.
func (r *Reranker) Provider() string { return r.provider }

// Score returns one probability per passage, batched.
func (r *Reranker) Score(ctx context.Context, query string, passages []string) ([]float32, error) {
	pairs := make([][]int64, len(passages))
	for i, p := range passages {
		ids, err := r.buildPair(query, p)
		if err != nil {
			return nil, err
		}
		pairs[i] = ids
	}
	out := make([]float32, len(pairs))
	for start := 0; start < len(pairs); start += r.batch {
		end := start + r.batch
		if end > len(pairs) {
			end = len(pairs)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		scores, err := r.runBatch(pairs[start:end])
		if err != nil {
			return nil, err
		}
		copy(out[start:end], scores)
	}
	return out, nil
}

// buildPair assembles <s> query </s> </s> passage </s> and truncates from the
// passage side, which is where the length lives.
func (r *Reranker) buildPair(query, passage string) ([]int64, error) {
	if r.format == "qwen3" {
		return r.buildQwenPair(query, passage)
	}
	q, err := r.encode(query)
	if err != nil {
		return nil, err
	}
	p, err := r.encode(passage)
	if err != nil {
		return nil, err
	}
	limit := r.pairLimit
	if limit <= 0 || limit > r.maxTokens {
		limit = r.maxTokens
	}
	budget := limit - len(q) - 4
	if budget < 16 {
		budget = 16
	}
	if len(p) > budget {
		p = p[:budget]
	}
	ids := make([]int64, 0, len(q)+len(p)+4)
	ids = append(ids, r.cls)
	ids = append(ids, q...)
	ids = append(ids, r.sep, r.sep)
	ids = append(ids, p...)
	ids = append(ids, r.sep)
	return ids, nil
}

// qwenPrefix and qwenSuffix are the ChatML wrapper Qwen3-Reranker was trained
// with. The seq-cls export folds the yes/no classifier into a single score
// head, so the whole prompt is plain text: no manual <s> pair is involved.
const (
	qwenPrefix = `<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
`
	qwenSuffix = `<|im_end|>
<|im_start|>assistant


`
	qwenDefaultInstruct = "Given a web search query, retrieve relevant passages that answer the query"
)

// buildQwenPair assembles the ChatML prompt and truncates the document, never
// the prefix/suffix: cutting the assistant tail would change the scoring head.
func (r *Reranker) buildQwenPair(query, passage string) ([]int64, error) {
	instruct := r.instruct
	if instruct == "" {
		instruct = qwenDefaultInstruct
	}
	head := qwenPrefix + "<Instruct>: " + instruct + "\n\n<Query>: " + query + "\n\n<Document>: "
	pre, err := r.encode(head)
	if err != nil {
		return nil, err
	}
	suf, err := r.encode(qwenSuffix)
	if err != nil {
		return nil, err
	}
	doc, err := r.encode(passage)
	if err != nil {
		return nil, err
	}
	limit := r.pairLimit
	if limit <= 0 || limit > r.maxTokens {
		limit = r.maxTokens
	}
	budget := limit - len(pre) - len(suf)
	if budget < 16 {
		budget = 16
	}
	if len(doc) > budget {
		doc = doc[:budget]
	}
	ids := make([]int64, 0, len(pre)+len(doc)+len(suf))
	ids = append(ids, pre...)
	ids = append(ids, doc...)
	ids = append(ids, suf...)
	return ids, nil
}
func (r *Reranker) runBatch(pairs [][]int64) ([]float32, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	rows := len(pairs)
	maxLen := 0
	for _, p := range pairs {
		if len(p) > maxLen {
			maxLen = len(p)
		}
	}
	ids := make([]int64, rows*maxLen)
	mask := make([]int64, rows*maxLen)
	for i, p := range pairs {
		for j, id := range p {
			ids[i*maxLen+j] = id
			mask[i*maxLen+j] = 1
		}
	}
	idsTensor, err := ort.NewTensor(ort.NewShape(int64(rows), int64(maxLen)), ids)
	if err != nil {
		return nil, err
	}
	defer idsTensor.Destroy()
	maskTensor, err := ort.NewTensor(ort.NewShape(int64(rows), int64(maxLen)), mask)
	if err != nil {
		return nil, err
	}
	defer maskTensor.Destroy()

	outputs := make([]ort.Value, 1)
	if err := r.session.Run([]ort.Value{idsTensor, maskTensor}, outputs); err != nil {
		return nil, fmt.Errorf("rerank: run: %w", err)
	}
	logits, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, errors.New("rerank: unexpected output type")
	}
	defer logits.Destroy()
	data := logits.GetData()
	if len(data) < rows {
		return nil, fmt.Errorf("rerank: output has %d values for %d pairs", len(data), rows)
	}
	out := make([]float32, rows)
	for i := 0; i < rows; i++ {
		out[i] = float32(1 / (1 + math.Exp(-float64(data[i*1]))))
	}
	return out, nil
}

func (r *Reranker) encode(text string) ([]int64, error) {
	enc, err := r.tok.EncodeWithOptionsErr(text, false)
	if err != nil {
		return nil, fmt.Errorf("rerank: tokenize: %w", err)
	}
	out := make([]int64, len(enc.IDs))
	for i, id := range enc.IDs {
		out[i] = int64(id)
	}
	return out, nil
}

// specialIDs reads the ids of <s> and </s>. Tokenizers come in two shapes:
// Unigram stores vocab as a list of [token, score] pairs (the index is the
// id), BPE stores a token->id map. Added tokens are checked first because
// some exports only declare the markers there.
func specialIDs(path string) ([2]int64, error) {
	var ids [2]int64
	var found [2]bool
	data, err := os.ReadFile(path)
	if err != nil {
		return ids, err
	}
	var doc struct {
		AddedTokens []struct {
			ID      int64  `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
		Model struct {
			Vocab json.RawMessage `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		return ids, fmt.Errorf("rerank: parse tokenizer: %w", err)
	}
	set := func(token string, id int64) {
		switch token {
		case "<s>":
			ids[0], found[0] = id, true
		case "</s>":
			ids[1], found[1] = id, true
		}
	}
	for _, t := range doc.AddedTokens {
		set(t.Content, t.ID)
	}
	vocab := strings.TrimSpace(string(doc.Model.Vocab))
	switch {
	case strings.HasPrefix(vocab, "["):
		var pairs []json.RawMessage
		if err := json.Unmarshal(doc.Model.Vocab, &pairs); err != nil {
			return ids, fmt.Errorf("rerank: parse unigram vocab: %w", err)
		}
		for i, raw := range pairs {
			var pair []json.RawMessage
			if err := json.Unmarshal(raw, &pair); err != nil || len(pair) == 0 {
				continue
			}
			var token string
			if err := json.Unmarshal(pair[0], &token); err != nil {
				continue
			}
			set(token, int64(i))
		}
	case strings.HasPrefix(vocab, "{"):
		var table map[string]int64
		if err := json.Unmarshal(doc.Model.Vocab, &table); err != nil {
			return ids, fmt.Errorf("rerank: parse bpe vocab: %w", err)
		}
		for token, id := range table {
			set(token, id)
		}
	default:
		return ids, errors.New("rerank: tokenizer has no vocabulary")
	}
	if !found[0] || !found[1] {
		return ids, fmt.Errorf("rerank: tokenizer does not declare <s> and </s> (found %v/%v)", found[0], found[1])
	}
	return ids, nil
}

// attachProvider mirrors the embedding policy: a discrete GPU first, then the
// integrated one, then CPU.
func attachProvider(so *ort.SessionOptions, opts Options) string {
	if opts.Device == "cpu" || opts.Device == "" {
		return "CPU"
	}
	gpus := resources.Detect()
	if len(gpus) == 0 {
		return "CPU"
	}
	var gpu resources.GPU
	if opts.GPUIndex >= 0 && opts.GPUIndex < len(gpus) {
		gpu = gpus[opts.GPUIndex]
	} else if best, ok := resources.Best(gpus); ok {
		gpu = best
	}
	switch gpu.Provider() {
	case "CUDA":
		if err := so.AppendExecutionProvider("CUDAExecutionProvider", map[string]string{"device_id": strconv.Itoa(gpu.Index)}); err == nil {
			return "CUDA"
		}
	case "ROCM":
		if err := so.AppendExecutionProvider("ROCMExecutionProvider", nil); err == nil {
			return "ROCM"
		}
	case "OpenVINO":
		if err := so.AppendExecutionProvider("OpenVINOExecutionProvider", map[string]string{"device_type": "GPU"}); err == nil {
			return "OpenVINO"
		}
	}
	return "CPU"
}
