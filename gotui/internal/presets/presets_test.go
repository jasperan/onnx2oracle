package presets

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Shaped like `onnx2oracle presets --json`: rerankers carry null dims/pooling.
const sample = `[
  {"name": "all-MiniLM-L6-v2", "hf_repo": "sentence-transformers/all-MiniLM-L6-v2", "task": "embedding",
   "pooling": "mean", "dims": 384, "max_length": 256, "normalize": true,
   "oracle_name": "ALL_MINILM_L6_V2", "approx_size_mb": 90},
  {"name": "bge-small-en-v1.5", "hf_repo": "BAAI/bge-small-en-v1.5", "task": "embedding",
   "pooling": "cls", "dims": 384, "max_length": 512, "normalize": true,
   "oracle_name": "BGE_SMALL_EN_V1_5", "approx_size_mb": 130},
  {"name": "ms-marco-MiniLM-L-12-v2", "hf_repo": "cross-encoder/ms-marco-MiniLM-L-12-v2", "task": "reranker",
   "pooling": null, "dims": null, "max_length": 512, "normalize": false,
   "oracle_name": "MS_MARCO_MINILM_L_12_V2", "approx_size_mb": 130},
  {"name": "ms-marco-MiniLM-L-6-v2", "hf_repo": "cross-encoder/ms-marco-MiniLM-L-6-v2", "task": "reranker",
   "max_length": 512, "oracle_name": "MS_MARCO_MINILM_L_6_V2"}
]`

func TestParseDecodesEmbeddingPreset(t *testing.T) {
	got, err := Parse(sample)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(got) != 4 {
		t.Fatalf("parsed %d presets, want 4", len(got))
	}
	want := Preset{Name: "all-MiniLM-L6-v2", Task: "embedding", Dims: 384, Pooling: "mean", SizeMB: 90,
		OracleName: "ALL_MINILM_L6_V2"}
	if got[0] != want {
		t.Errorf("first preset = %+v, want %+v", got[0], want)
	}
}

// A long name must come through intact: it is passed to the CLI as an argument.
func TestParseKeepsLongNamesIntact(t *testing.T) {
	got, err := Parse(sample)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got[2].Name != "ms-marco-MiniLM-L-12-v2" {
		t.Errorf("name = %q, want the untruncated name", got[2].Name)
	}
}

// A reranker has no dims or pooling, whether the CLI sends null or omits the key.
func TestParseRerankerWithoutDimsOrPooling(t *testing.T) {
	got, err := Parse(sample)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	for _, i := range []int{2, 3} {
		r := got[i]
		if r.Task != "reranker" || r.Dims != 0 || r.Pooling != "" {
			t.Errorf("preset %d = %+v, want a reranker with dims 0 and empty pooling", i, r)
		}
	}
	if got[2].SizeMB != 130 || got[2].OracleName != "MS_MARCO_MINILM_L_12_V2" {
		t.Errorf("reranker = %+v, want size and oracle name preserved", got[2])
	}
}

func TestParseRejectsBadInput(t *testing.T) {
	cases := map[string]string{
		"empty":         "",
		"not json":      "┃ Name ┃ Task ┃",
		"empty array":   "[]",
		"object":        `{"name": "x"}`,
		"missing name":  `[{"task": "embedding"}]`,
		"missing task":  `[{"name": "x"}]`,
		"wrong dims":    `[{"name": "x", "task": "embedding", "dims": "384"}]`,
		"trailing text": `[{"name": "x", "task": "embedding"}] extra`,
	}
	for name, in := range cases {
		if _, err := Parse(in); err == nil {
			t.Errorf("Parse(%s) = nil error, want a rejection so a contract change is loud", name)
		}
	}
}

// Load must run the binary with `presets --json` and decode its stdout.
func TestLoadRunsPresetsJSON(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "fake-onnx2oracle")
	script := "#!/bin/sh\n" +
		"if [ \"$1\" != presets ] || [ \"$2\" != --json ]; then echo \"bad args: $*\" >&2; exit 2; fi\n" +
		"cat <<'EOF'\n" + sample + "\nEOF\n"
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := Load(bin, 10*time.Second)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(got) != 4 || got[1].Name != "bge-small-en-v1.5" || got[1].Pooling != "cls" {
		t.Errorf("Load = %+v, want the decoded fixture", got)
	}
}

func TestLoadSurfacesCLIFailure(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "fake-onnx2oracle")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\necho 'Error: No such option: --json' >&2\nexit 2\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	_, err := Load(bin, 10*time.Second)
	if err == nil || !strings.Contains(err.Error(), "No such option: --json") {
		t.Fatalf("Load error = %v, want the CLI's stderr message", err)
	}
}

func TestLabelNamesTaskAndSize(t *testing.T) {
	embed := Preset{Name: "e", Task: "embedding", Dims: 384, Pooling: "mean", SizeMB: 90}
	if got, want := embed.Label(), "e — embedding, 384 dims, mean, ~90 MB"; got != want {
		t.Errorf("Label() = %q, want %q", got, want)
	}
	rk := Preset{Name: "r", Task: "reranker", SizeMB: 90}
	if got, want := rk.Label(), "r — reranker, ~90 MB"; got != want {
		t.Errorf("Label() = %q, want %q", got, want)
	}
	unsized := Preset{Name: "r", Task: "reranker"}
	if got, want := unsized.Label(), "r — reranker"; got != want {
		t.Errorf("Label() = %q, want %q (no bogus ~0 MB)", got, want)
	}
}
