// Package presets reads the curated model registry that the onnx2oracle CLI
// exposes through `onnx2oracle presets --json`.
//
// The names are never invented here: they come from the CLI's own registry, which
// is the same source of truth a user would read. Parse is pure and unit-tested
// against fixed JSON so the decoding contract does not depend on a live run.
package presets

import (
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// Preset is one entry of `onnx2oracle presets --json`.
type Preset struct {
	Name       string
	Task       string // "embedding" or "reranker"
	Dims       int    // 0 when the CLI reports no value (rerankers)
	Pooling    string // "" when not applicable
	SizeMB     int    // 0 when the CLI does not report an approximate size
	OracleName string
}

// Label is the single-line form used in a huh Select option.
func (p Preset) Label() string {
	size := ""
	if p.SizeMB > 0 {
		size = fmt.Sprintf(", ~%d MB", p.SizeMB)
	}
	if p.Task == "embedding" {
		return fmt.Sprintf("%s — %s, %d dims, %s%s", p.Name, p.Task, p.Dims, p.Pooling, size)
	}
	return fmt.Sprintf("%s — %s%s", p.Name, p.Task, size)
}

// wire mirrors one JSON object. Dims and Pooling are pointers because a
// reranker has neither and the CLI may emit null or omit the key.
type wire struct {
	Name       string  `json:"name"`
	Task       string  `json:"task"`
	Dims       *int    `json:"dims"`
	Pooling    *string `json:"pooling"`
	SizeMB     int     `json:"approx_size_mb"`
	OracleName string  `json:"oracle_name"`
}

// Parse decodes the JSON array printed by `onnx2oracle presets --json`.
func Parse(out string) ([]Preset, error) {
	var items []wire
	if err := json.Unmarshal([]byte(out), &items); err != nil {
		return nil, fmt.Errorf("decoding presets JSON: %w", err)
	}
	got := make([]Preset, 0, len(items))
	for i, it := range items {
		if it.Name == "" || it.Task == "" {
			return nil, fmt.Errorf("preset %d has no name or task", i)
		}
		p := Preset{Name: it.Name, Task: it.Task, SizeMB: it.SizeMB, OracleName: it.OracleName}
		if it.Dims != nil {
			p.Dims = *it.Dims
		}
		if it.Pooling != nil {
			p.Pooling = *it.Pooling
		}
		got = append(got, p)
	}
	if len(got) == 0 {
		return nil, errors.New("the CLI reported no presets")
	}
	return got, nil
}

// Load runs `onnx2oracle presets --json` and decodes the result.
func Load(binary string, timeout time.Duration) ([]Preset, error) {
	cmd := exec.Command(binary, "presets", "--json")
	var out, stderr strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &stderr
	if err := runWithTimeout(cmd, timeout); err != nil {
		if msg := strings.TrimSpace(stderr.String()); msg != "" {
			err = fmt.Errorf("%w: %s", err, lastLine(msg))
		}
		return nil, fmt.Errorf("running %s presets --json: %w", binary, err)
	}
	return Parse(out.String())
}

func lastLine(s string) string {
	if i := strings.LastIndex(s, "\n"); i >= 0 {
		return strings.TrimSpace(s[i+1:])
	}
	return s
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
