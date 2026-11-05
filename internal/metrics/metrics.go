// Package metrics is a tiny Prometheus-compatible registry.
//
// It exists to avoid pulling a client library for four counters: the exposition
// format is a few lines of text and the rest is state we already have.
package metrics

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

type observation struct {
	sum   float64
	count uint64
}

// Registry holds counters, gauges and summaries.
type Registry struct {
	mu     sync.Mutex
	values map[string]float64
	obs    map[string]*observation
	kinds  map[string]string // name -> counter|gauge|summary
}

// New returns an empty registry.
func New() *Registry {
	return &Registry{
		values: map[string]float64{},
		obs:    map[string]*observation{},
		kinds:  map[string]string{},
	}
}

// Inc adds one to a counter.
func (r *Registry) Inc(name string, labels Labels) { r.Add(name, 1, labels) }

// Add adds v to a counter.
func (r *Registry) Add(name string, v float64, labels Labels) {
	key := keyOf(name, labels)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.kinds[name] = "counter"
	r.values[key] += v
}

// Set stores a gauge value.
func (r *Registry) Set(name string, v float64, labels Labels) {
	key := keyOf(name, labels)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.kinds[name] = "gauge"
	r.values[key] = v
}

// Observe records one duration in a summary (name_sum / name_count).
func (r *Registry) Observe(name string, v float64, labels Labels) {
	key := keyOf(name, labels)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.kinds[name] = "summary"
	o := r.obs[key]
	if o == nil {
		o = &observation{}
		r.obs[key] = o
	}
	o.sum += v
	o.count++
}

// Labels are rendered as name="value" (sorted for stable output).
type Labels map[string]string

func keyOf(name string, labels Labels) string {
	if len(labels) == 0 {
		return name
	}
	keys := make([]string, 0, len(labels))
	for k := range labels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	b.WriteString(name)
	b.WriteString("{")
	for i, k := range keys {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, `%s=%q`, k, labels[k])
	}
	b.WriteString("}")
	return b.String()
}

// Render returns the Prometheus text exposition.
func (r *Registry) Render() string {
	r.mu.Lock()
	defer r.mu.Unlock()

	byName := map[string][]string{}
	for key := range r.values {
		name := key
		if i := strings.IndexByte(key, '{'); i >= 0 {
			name = key[:i]
		}
		byName[name] = append(byName[name], key)
	}
	for key := range r.obs {
		name := key
		if i := strings.IndexByte(key, '{'); i >= 0 {
			name = key[:i]
		}
		byName[name] = append(byName[name], key)
	}

	names := make([]string, 0, len(byName))
	for n := range byName {
		names = append(names, n)
	}
	sort.Strings(names)

	var b strings.Builder
	for _, name := range names {
		kind := r.kinds[name]
		switch kind {
		case "summary":
			fmt.Fprintf(&b, "# TYPE %s summary\n", name)
			keys := byName[name]
			sort.Strings(keys)
			for _, key := range keys {
				o := r.obs[key]
				fmt.Fprintf(&b, "%s_sum %g\n%s_count %d\n", key, o.sum, key, o.count)
			}
		default:
			fmt.Fprintf(&b, "# TYPE %s %s\n", name, kind)
			keys := byName[name]
			sort.Strings(keys)
			for _, key := range keys {
				fmt.Fprintf(&b, "%s %g\n", key, r.values[key])
			}
		}
	}
	return b.String()
}
