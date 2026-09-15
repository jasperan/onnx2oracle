package run

import (
	"strings"
	"testing"

	"github.com/jasperan/onnx2oracle/gotui/internal/dsn"
)

func testDSN() dsn.DSN {
	return dsn.DSN{User: "system", Password: "topsecret", Host: "localhost", Port: 1521, Service: "FREEPDB1"}
}

// The central safety property: the password must never be reachable from argv.
func TestPasswordIsNeverInArgv(t *testing.T) {
	d := testDSN()
	invs := []Invocation{
		Load(d, "onnx2oracle", LoadRequest{Preset: "all-MiniLM-L6-v2", Force: true}),
		Preflight(d, "onnx2oracle"),
		Verify(d, "onnx2oracle", "ALL_MINILM_L6_V2"),
		Rerank(d, "onnx2oracle", "q", "MS_MARCO_MINILM_L_6_V2", []string{"doc one", "doc two"}),
	}
	for _, inv := range invs {
		joined := strings.Join(inv.Argv, " ")
		if strings.Contains(joined, d.Password) {
			t.Errorf("argv leaks the password: %v", inv.Argv)
		}
		for _, a := range inv.Argv {
			if strings.Contains(a, "--password") || strings.Contains(a, "-p=") {
				t.Errorf("argv carries a password flag: %q", a)
			}
		}
	}
}

// And it must be present in the environment, or the run has no credentials.
func TestPasswordIsInEnvViaOracleDSN(t *testing.T) {
	d := testDSN()
	inv := Preflight(d, "onnx2oracle")
	var found string
	for _, kv := range inv.Env {
		if strings.HasPrefix(kv, dsn.EnvDSN+"=") {
			found = kv
		}
	}
	if found == "" {
		t.Fatalf("env has no %s: %v", dsn.EnvDSN, inv.Env)
	}
	if !strings.Contains(found, d.Password) {
		t.Errorf("%s = %q, want it to carry the credentials", dsn.EnvDSN, found)
	}
}

// A pre-existing ORACLE_DSN must be replaced, not duplicated: two entries leave
// the winning one up to the runtime and can silently connect elsewhere.
func TestEnvReplacesExistingOracleDSN(t *testing.T) {
	t.Setenv(dsn.EnvDSN, "someone/pw@elsewhere:1521/OTHER")
	inv := Preflight(testDSN(), "onnx2oracle")
	n := 0
	for _, kv := range inv.Env {
		if strings.HasPrefix(kv, dsn.EnvDSN+"=") {
			n++
			if strings.Contains(kv, "elsewhere") {
				t.Errorf("stale %s survived: %q", dsn.EnvDSN, kv)
			}
		}
	}
	if n != 1 {
		t.Errorf("found %d %s entries, want exactly 1", n, dsn.EnvDSN)
	}
}

// Flag spellings must match what `onnx2oracle load --help` prints.
func TestLoadArgvUsesRealFlagSpellings(t *testing.T) {
	yes := true
	got := Load(testDSN(), "onnx2oracle", LoadRequest{
		Preset:    "bge-small-en-v1.5",
		Name:      "MY_MODEL",
		Task:      "embedding",
		Pooling:   "cls",
		Dims:      384,
		MaxLength: 512,
		NormaliZe: &yes,
		Force:     true,
		CacheDir:  "/tmp/hf",
	})
	want := []string{
		"onnx2oracle", "load", "bge-small-en-v1.5",
		"--name", "MY_MODEL",
		"--task", "embedding",
		"--pooling", "cls",
		"--dims", "384",
		"--max-length", "512",
		"--normalize",
		"--force",
		"--cache-dir", "/tmp/hf",
	}
	if strings.Join(got.Argv, " ") != strings.Join(want, " ") {
		t.Fatalf("argv =\n  %v\nwant\n  %v", got.Argv, want)
	}
}

// typer's boolean pair must emit the negative spelling, not "false".
func TestLoadEmitsNoNormalize(t *testing.T) {
	no := false
	got := Load(testDSN(), "onnx2oracle", LoadRequest{Preset: "all-MiniLM-L6-v2", NormaliZe: &no})
	if !contains(got.Argv, "--no-normalize") {
		t.Errorf("argv = %v, want --no-normalize", got.Argv)
	}
	if contains(got.Argv, "--normalize") {
		t.Errorf("argv = %v, must not also carry --normalize", got.Argv)
	}
}

// Defaults must not be sent as flags: overriding --max-length with a value the
// user never chose is how a front-end silently changes the model it loads.
func TestLoadOmitsUnsetOptions(t *testing.T) {
	got := Load(testDSN(), "onnx2oracle", LoadRequest{Preset: "all-MiniLM-L6-v2"})
	want := "onnx2oracle load all-MiniLM-L6-v2"
	if strings.Join(got.Argv, " ") != want {
		t.Fatalf("argv = %q, want %q", strings.Join(got.Argv, " "), want)
	}
}

func TestRerankRepeatsDocFlag(t *testing.T) {
	got := Rerank(testDSN(), "onnx2oracle", "what is a database?", "MS_MARCO_MINILM_L_6_V2",
		[]string{"first passage", "second passage"})
	want := "onnx2oracle rerank --query what is a database? --doc first passage --doc second passage --name MS_MARCO_MINILM_L_6_V2"
	if strings.Join(got.Argv, " ") != want {
		t.Fatalf("argv = %q, want %q", strings.Join(got.Argv, " "), want)
	}
}

func TestVerifyOmitsNameWhenEmpty(t *testing.T) {
	got := Verify(testDSN(), "onnx2oracle", "")
	if strings.Join(got.Argv, " ") != "onnx2oracle verify" {
		t.Fatalf("argv = %v, want a bare verify", got.Argv)
	}
}

func TestRedactHidesPasswordInRenderedEnv(t *testing.T) {
	d := testDSN()
	line := dsn.EnvDSN + "=" + d.String()
	if got := Redact(line, d); strings.Contains(got, d.Password) {
		t.Errorf("Redact(%q) = %q, want the password masked", line, got)
	}
}

func contains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
