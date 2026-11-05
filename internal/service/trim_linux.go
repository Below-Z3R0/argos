//go:build linux && cgo

package service

/*
#include <malloc.h>
*/
import "C"

// trimHeap returns free malloc arenas to the OS. Without it, unloading the
// models leaves RSS high even though the memory is reusable by the process.
func trimHeap() { C.malloc_trim(0) }
