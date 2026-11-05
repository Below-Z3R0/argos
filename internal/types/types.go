// Package types holds the data structures shared across packages. It has no
// dependencies on the rest of the project, which keeps the dependency graph
// acyclic.
package types

// Point is one vector plus its payload, independent of the store driver.
type Point struct {
	ID      string
	Vector  []float32
	Payload map[string]any
}

// Hit is a search result, independent of the store driver.
type Hit struct {
	ID      string
	Score   float32
	Payload map[string]any
}
