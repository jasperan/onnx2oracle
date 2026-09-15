// Package presets reads the curated model registry that the onnx2oracle CLI
// exposes through `onnx2oracle presets`.
//
// The names are never invented here: they come from the CLI's own table, which
// is the same source of truth a user would read. Parse is pure and unit-tested
// against a fixed sample so the parsing contract does not depend on a live run.
package presets

import (
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// Preset is one row of `onnx2oracle presets`.
type Preset struct {
	Name       string
	Task       string // "embedding" or "reranker"
	Dims       int    // 0 when the CLI prints the not-applicable placeholder
	Pooling    string // "" when not applicable
	SizeMB     int
	OracleName string
}

// Label is the single-line form used in a huh Select option.
func (p Preset) Label() string {
	if p.Task == "embedding" {
		return fmt.Sprintf("%s — %s, %d dims, %s, ~%d MB", p.Name, p.Task, p.Dims, p.Pooling, p.SizeMB)
	}
	return fmt.Sprintf("%s — %s, ~%d MB", p.Name, p.Task, p.SizeMB)
}

// notApplicable is the placeholder the CLI prints for fields a reranker has no
// value for. Both spellings appear because rich renders an em dash.
const notApplicable = "—"

// Parse reads the CLI's table. It is deliberately tolerant of the box drawing
// characters changing: a row is any line with at least six '│' separated cells.
func Parse(out string) ([]Preset, error) {
	var got []Preset
	for _, raw := range strings.Split(out, "\n") {
		if strings.Count(raw, "│") < 6 {
			continue
		}
		cells := splitCells(raw)
		if len(cells) < 6 {
			continue
		}
		name := cells[0]
		if name == "" || strings.EqualFold(name, "Name") {
			continue
		}
		p := Preset{
			Name:       name,
			Task:       cells[1],
			Pooling:    naToEmpty(cells[3]),
			OracleName: cells[5],
		}
		p.Dims = naToInt(cells[2])
		p.SizeMB = naToInt(cells[4])
		if p.Task == "" {
			continue
		}
		got = append(got, p)
	}
	if len(got) == 0 {
		return nil, errors.New("no presets parsed: the table layout changed")
	}
	return got, nil
}

func splitCells(line string) []string {
	parts := strings.Split(line, "│")
	// The table is bordered, so the first and last cells are empty padding.
	if len(parts) > 1 {
		parts = parts[1 : len(parts)-1]
	}
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		out = append(out, strings.TrimSpace(p))
	}
	return out
}

func naToEmpty(s string) string {
	if s == notApplicable || s == "--" || s == "-" {
		return ""
	}
	return s
}

func naToInt(s string) int {
	n, err := strconv.Atoi(naToEmpty(s))
	if err != nil {
		return 0
	}
	return n
}

// Load runs `onnx2oracle presets` and parses the result.
//
// COLUMNS is widened because rich truncates cells to the terminal width, which
// would silently turn "all-MiniLM-L12-v2" into "all-MiniLM-L12…" and make the
// preset name unusable as an argument.
func Load(binary string, timeout time.Duration) ([]Preset, error) {
	cmd := exec.Command(binary, "presets")
	cmd.Env = append(cmd.Environ(), "COLUMNS=200")
	var out strings.Builder
	cmd.Stdout = &out
	if err := runWithTimeout(cmd, timeout); err != nil {
		return nil, fmt.Errorf("running %s presets: %w", binary, err)
	}
	return Parse(out.String())
}

func runWithTimeout(cmd *exec.Cmd, timeout time.Duration) error {
	if timeout <= 0 {
		return cmd.Run()
	}
	done := make(chan error, 1)
	if err := cmd.Start(); err != nil {
		return err
	}
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		return err
	case <-time.After(timeout):
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		<-done
		return fmt.Errorf("timed out after %s", timeout)
	}
}
