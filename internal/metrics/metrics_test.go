package metrics

import (
	"strings"
	"testing"
)

func TestRenderCounterGaugeSummary(t *testing.T) {
	r := New()
	l := Labels{"model": "qwen3", "state": "ready"}
	r.Inc("argos_model_load_total", l)
	r.Add("argos_model_load_total", 2, l)
	r.Set("argos_models_loaded", 1, Labels{"kind": "embed"})
	r.Observe("argos_search_seconds", 0.05, nil)
	r.Observe("argos_search_seconds", 0.15, nil)

	out := r.Render()
	for _, want := range []string{
		`argos_model_load_total{model="qwen3",state="ready"} 3`,
		`argos_models_loaded{kind="embed"} 1`,
		"argos_search_seconds_sum 0.2",
		"argos_search_seconds_count 2",
		"# TYPE argos_search_seconds summary",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
}

func TestLabelOrderIsStable(t *testing.T) {
	r := New()
	r.Inc("m", Labels{"b": "2", "a": "1"})
	if !strings.Contains(r.Render(), `m{a="1",b="2"} 1`) {
		t.Fatalf("labels not sorted:\n%s", r.Render())
	}
}
