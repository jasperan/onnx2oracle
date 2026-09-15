package tui

import (
	"context"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"github.com/jasperan/onnx2oracle/gotui/internal/run"
)

// ShellExec runs the real onnx2oracle CLI.
//
// Every run is bounded: without a timeout an unreachable Oracle would hang the
// TUI forever, which is the failure mode this front-end exists to avoid.
type ShellExec struct {
	// Dir is the working directory for the child; empty means inherit.
	Dir string
}

// Run executes the invocation and returns its combined output.
func (e ShellExec) Run(inv run.Invocation, timeout time.Duration) (string, error) {
	if len(inv.Argv) == 0 {
		return "", fmt.Errorf("empty invocation")
	}
	ctx := context.Background()
	var cancel context.CancelFunc
	if timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	cmd := exec.CommandContext(ctx, inv.Argv[0], inv.Argv[1:]...)
	cmd.Env = inv.Env
	cmd.Dir = e.Dir
	var out strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &out

	err := cmd.Run()
	text := out.String()
	if ctx.Err() == context.DeadlineExceeded {
		return text, fmt.Errorf("timed out after %s", timeout)
	}
	if err != nil {
		if _, lookErr := exec.LookPath(inv.Argv[0]); lookErr != nil {
			return text, fmt.Errorf("%s not found on PATH: %w", inv.Argv[0], lookErr)
		}
		return text, fmt.Errorf("%s", firstLine(text, err))
	}
	return text, nil
}

// errorLine matches a real exception header ("DatabaseError: ...",
// "TypeError: ...") and deliberately not a traceback source line such as
// "raise TypeError(message)", which is a code frame and not the failure.
var errorLine = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)\s*:`)

// firstLine prefers the CLI's own error message, which is far more useful than
// "exit status 1". Rich-formatted output is box drawn, so borders are skipped,
// and a real exception header wins over a status line.
func firstLine(output string, err error) string {
	fallback := ""
	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		s := strings.TrimSpace(stripANSI(line))
		s = strings.Trim(s, "│┃ ")
		s = strings.TrimSpace(s)
		if s == "" || strings.IndexFunc(s, isWordRune) < 0 {
			continue
		}
		if errorLine.MatchString(s) {
			return s
		}
		if fallback == "" {
			fallback = s
		}
	}
	if fallback != "" {
		return fallback
	}
	return err.Error()
}

func isWordRune(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')
}

func stripANSI(s string) string {
	var b strings.Builder
	esc := false
	for _, r := range s {
		switch {
		case esc:
			if r == 'm' {
				esc = false
			}
		case r == 0x1b:
			esc = true
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// DefaultTimeout is generous: an ONNX export plus a database insert is slow, but
// it still has to end.
const DefaultTimeout = 10 * time.Minute
