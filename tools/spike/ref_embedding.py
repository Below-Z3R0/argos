"""Reference implementation for the embedding spike.

Loads the same ONNX export with onnxruntime, applies last-token pooling and
L2 normalization, and compares against the vectors produced by the Go code.

Run with the RAG venv:  ~/Documents/Knowledge/.rag/venv/bin/python ref.py
"""

import json
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else None
GO_VECTORS = sys.argv[2] if len(sys.argv) > 2 else "/tmp/go-vectors.json"

if not MODEL_DIR:
    raise SystemExit("uso: ref.py <dir-del-modelo> [go-vectors.json]")

payload = json.load(open(GO_VECTORS))
texts = payload["texts"]
go_vecs = np.asarray(payload["vectors"], dtype=np.float32)

tk = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
sess = ort.InferenceSession(f"{MODEL_DIR}/model.onnx", providers=["CPUExecutionProvider"])

inputs = {i.name: i for i in sess.get_inputs()}
kv_names = [n for n in inputs if n.startswith("past_key_values.")]

out = []
for text in texts:
    enc = tk.encode(text)
    ids = np.asarray([enc.ids], dtype=np.int64)
    n = ids.shape[1]
    feeds = {
        "input_ids": ids,
        "attention_mask": np.ones((1, n), dtype=np.int64),
        "position_ids": np.arange(n, dtype=np.int64)[None, :],
    }
    for name in kv_names:
        shape = inputs[name].shape
        heads = shape[1] if isinstance(shape[1], int) else 8
        head_dim = shape[3] if isinstance(shape[3], int) else 128
        feeds[name] = np.zeros((1, heads, 0, head_dim), dtype=np.float32)

    hidden = sess.run(["last_hidden_state"], feeds)[0][0]
    vec = hidden[-1].astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    out.append(vec)

py_vecs = np.asarray(out, dtype=np.float32)

print(f"dim={py_vecs.shape[1]} n={py_vecs.shape[0]}")
worst = 1.0
for i in range(len(texts)):
    c = float(np.dot(go_vecs[i], py_vecs[i]))
    worst = min(worst, c)
    print(f"cos(go[{i}], py[{i}]) = {c:.6f}")
print(f"peor coincidencia = {worst:.6f} (umbral 0.999)")
print("RESULTADO:", "OK" if worst >= 0.999 else "FALLA")
