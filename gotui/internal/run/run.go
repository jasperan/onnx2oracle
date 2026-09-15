// Package run builds the argv/env for invoking the real onnx2oracle CLI.
//
// The subcommand and flag spellings here are taken from src/onnx2oracle/cli.py
// and cross-checked against `onnx2oracle load --help`. Nothing is guessed.
//
// The password rides in the environment (ORACLE_DSN), never in argv: connection.py's
// resolve_dsn() reads ORACLE_DSN before it ever prompts, so a full DSN in the
// environment both authenticates and suppresses the Python-side interactive prompt.
// argv is therefore safe to show in a process list, a log, or the TUI.
package run

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/jasperan/onnx2oracle/gotui/internal/dsn"
)

// Invocation is a ready-to-exec subprocess call.
type Invocation struct {
	Argv []string
	Env  []string
}

// CommandLine renders argv for display. It can never contain the password.
func (i Invocation) CommandLine() string { return strings.Join(i.Argv, " ") }

// envWithDSN returns the current environment plus the resolved connection.
func envWithDSN(d dsn.DSN) []string {
	env := os.Environ()
	out := make([]string, 0, len(env)+1)
	for _, kv := range env {
		// Replace rather than append: a stale ORACLE_DSN would win by position in
		// some shells and silently connect somewhere else.
		if strings.HasPrefix(kv, dsn.EnvDSN+"=") {
			continue
		}
		out = append(out, kv)
	}
	return append(out, fmt.Sprintf("%s=%s", dsn.EnvDSN, d.String()))
}

// LoadRequest mirrors the options of `onnx2oracle load`.
type LoadRequest struct {
	Preset    string
	Name      string
	Task      string
	Pooling   string
	Dims      int
	MaxLength int
	NormaliZe *bool
	Force     bool
	CacheDir  string
}

// Load builds `onnx2oracle load [PRESET] [--flags]`.
func Load(d dsn.DSN, binary string, r LoadRequest) Invocation {
	argv := []string{binary, "load"}
	if r.Preset != "" {
		argv = append(argv, r.Preset)
	}
	if r.Name != "" {
		argv = append(argv, "--name", r.Name)
	}
	if r.Task != "" {
		argv = append(argv, "--task", r.Task)
	}
	if r.Pooling != "" {
		argv = append(argv, "--pooling", r.Pooling)
	}
	if r.Dims > 0 {
		argv = append(argv, "--dims", strconv.Itoa(r.Dims))
	}
	if r.MaxLength > 0 {
		argv = append(argv, "--max-length", strconv.Itoa(r.MaxLength))
	}
	if r.NormaliZe != nil {
		// cli.py declares this as a typer --normalize/--no-normalize pair.
		if *r.NormaliZe {
			argv = append(argv, "--normalize")
		} else {
			argv = append(argv, "--no-normalize")
		}
	}
	if r.Force {
		argv = append(argv, "--force")
	}
	if r.CacheDir != "" {
		argv = append(argv, "--cache-dir", r.CacheDir)
	}
	return Invocation{Argv: argv, Env: envWithDSN(d)}
}

// Preflight builds `onnx2oracle preflight` (connection + privileges check).
func Preflight(d dsn.DSN, binary string) Invocation {
	return Invocation{Argv: []string{binary, "preflight"}, Env: envWithDSN(d)}
}

// Verify builds `onnx2oracle verify [--name NAME]`.
func Verify(d dsn.DSN, binary, name string) Invocation {
	argv := []string{binary, "verify"}
	if name != "" {
		argv = append(argv, "--name", name)
	}
	return Invocation{Argv: argv, Env: envWithDSN(d)}
}

// Rerank builds `onnx2oracle rerank --query Q --doc D... --name N`.
func Rerank(d dsn.DSN, binary, query, name string, docs []string) Invocation {
	argv := []string{binary, "rerank", "--query", query}
	for _, doc := range docs {
		argv = append(argv, "--doc", doc)
	}
	if name != "" {
		argv = append(argv, "--name", name)
	}
	return Invocation{Argv: argv, Env: envWithDSN(d)}
}

// Redact masks the password inside a rendered command line, for the rare case
// something prints the environment. argv itself never needs it.
func Redact(s string, d dsn.DSN) string {
	if d.Password == "" {
		return s
	}
	return strings.ReplaceAll(s, d.Password, "********")
}
