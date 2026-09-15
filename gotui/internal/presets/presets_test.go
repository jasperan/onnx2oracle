package presets

import "testing"

// A verbatim COLUMNS=200 capture of `onnx2oracle presets`, so the parser is
// pinned against the real table rather than a hand-written idealisation.
const sample = `                                      onnx2oracle presets
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name                    ┃ Task      ┃ Dims ┃ Pooling ┃ ~Size (MB) ┃ Oracle model name       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ all-MiniLM-L6-v2        │ embedding │  384 │ mean    │         90 │ ALL_MINILM_L6_V2        │
│ all-MiniLM-L12-v2       │ embedding │  384 │ mean    │        130 │ ALL_MINILM_L12_V2       │
│ all-mpnet-base-v2       │ embedding │  768 │ mean    │        420 │ ALL_MPNET_BASE_V2       │
│ bge-small-en-v1.5       │ embedding │  384 │ cls     │        130 │ BGE_SMALL_EN_V1_5       │
│ nomic-embed-text-v1     │ embedding │  768 │ mean    │        540 │ NOMIC_EMBED_TEXT_V1     │
│ ms-marco-MiniLM-L-6-v2  │ reranker  │    — │ —       │         90 │ MS_MARCO_MINILM_L_6_V2  │
│ ms-marco-MiniLM-L-12-v2 │ reranker  │    — │ —       │        130 │ MS_MARCO_MINILM_L_12_V2 │
└─────────────────────────┴───────────┴──────┴─────────┴────────────┴─────────────────────────┘
`

func TestParseRealTable(t *testing.T) {
	got, err := Parse(sample)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(got) != 7 {
		t.Fatalf("parsed %d presets, want 7", len(got))
	}

	first := got[0]
	if first.Name != "all-MiniLM-L6-v2" || first.Task != "embedding" || first.Dims != 384 ||
		first.Pooling != "mean" || first.SizeMB != 90 || first.OracleName != "ALL_MINILM_L6_V2" {
		t.Errorf("first preset = %+v, want the all-MiniLM-L6-v2 row verbatim", first)
	}

	// A long name must survive intact: truncation would make it unusable as an argument.
	last := got[6]
	if last.Name != "ms-marco-MiniLM-L-12-v2" {
		t.Errorf("last preset name = %q, want the untruncated name", last.Name)
	}
}

// A reranker has no dims or pooling, and the CLI prints an em dash. That must
// become zero/empty rather than a parse error or a bogus 0-dims embedding.
func TestParseRerankerPlaceholders(t *testing.T) {
	got, err := Parse(sample)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	reranker := got[5]
	if reranker.Task != "reranker" {
		t.Fatalf("row 5 task = %q, want reranker", reranker.Task)
	}
	if reranker.Dims != 0 || reranker.Pooling != "" {
		t.Errorf("reranker = %+v, want dims 0 and empty pooling for the em dash", reranker)
	}
	if reranker.SizeMB != 90 || reranker.OracleName != "MS_MARCO_MINILM_L_6_V2" {
		t.Errorf("reranker = %+v, want size and oracle name preserved", reranker)
	}
}

func TestParseRejectsEmptyInput(t *testing.T) {
	if _, err := Parse(""); err == nil {
		t.Error("Parse(\"\") = nil, want an error so a layout change is loud")
	}
}

func TestParseIgnoresHeaderAndBorders(t *testing.T) {
	got, err := Parse(sample)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	for _, p := range got {
		if p.Name == "Name" || p.Task == "Task" || p.Name == "" {
			t.Fatalf("header leaked into the result: %+v", p)
		}
	}
}

func TestLabelNamesTaskAndOracleName(t *testing.T) {
	embed := Preset{Name: "e", Task: "embedding", Dims: 384, Pooling: "mean", SizeMB: 90}
	if got := embed.Label(); got == "" {
		t.Error("Label() is empty")
	}
	rk := Preset{Name: "r", Task: "reranker", SizeMB: 90}
	if got := rk.Label(); got == "" {
		t.Error("Label() is empty for a reranker")
	}
}
