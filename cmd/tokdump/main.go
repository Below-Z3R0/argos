// Command tokdump prints token ids for a text, to compare tokenizers.
package main

import (
	"fmt"
	"os"

	"github.com/daulet/tokenizers"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "uso: tokdump <tokenizer.json> <texto>")
		os.Exit(2)
	}
	tk, err := tokenizers.FromFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer tk.Close()
	enc, err := tk.EncodeWithOptionsErr(os.Args[2], true)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(enc.IDs)
}
