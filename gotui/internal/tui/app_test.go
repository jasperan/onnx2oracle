package tui

import (
	"errors"
	"strings"
	"testing"
	"time"

	"charm.land/bubbles/v2/cursor"
	tea "charm.land/bubbletea/v2"

	"github.com/jasperan/onnx2oracle/gotui/internal/dsn"
	"github.com/jasperan/onnx2oracle/gotui/internal/presets"
	"github.com/jasperan/onnx2oracle/gotui/internal/run"
)

type fakeExec struct {
	gotInv run.Invocation
	out    string
	err    error
	calls  int
}

func (f *fakeExec) Run(inv run.Invocation, _ time.Duration) (string, error) {
	f.gotInv = inv
	f.calls++
	return f.out, f.err
}

func samplePresets() []presets.Preset {
	return []presets.Preset{
		{Name: "all-MiniLM-L6-v2", Task: "embedding", Dims: 384, Pooling: "mean", SizeMB: 90, OracleName: "ALL_MINILM_L6_V2"},
		{Name: "ms-marco-MiniLM-L-6-v2", Task: "reranker", SizeMB: 90, OracleName: "MS_MARCO_MINILM_L_6_V2"},
	}
}

func newModel(exec Executor) Model {
	return New(Options{
		Presets: samplePresets(),
		Exec:    exec,
		Binary:  "onnx2oracle",
		Timeout: time.Second,
		Seed:    dsn.DSN{User: "system", Password: "topsecret", Host: "localhost", Port: 1521, Service: "FREEPDB1"},
	})
}

// The forms cannot be driven headlessly, so StartLoad is where the answers-turn-
// into-an-invocation decision is pinned.
func TestStartLoadBuildsInvocationFromAnswers(t *testing.T) {
	fe := &fakeExec{out: "registered"}
	m := newModel(fe)
	m.ans.User = "system"
	m.ans.Password = "topsecret"
	m.ans.Host = "localhost"
	m.ans.Port = "1521"
	m.ans.Service = "FREEPDB1"
	m.ans.Preset = "all-MiniLM-L6-v2"
	m.ans.Force = true
	m.ans.Confirmed = true

	next, cmd, err := m.StartLoad()
	if err != nil {
		t.Fatalf("StartLoad: %v", err)
	}
	if fe.calls != 0 {
		t.Fatalf("StartLoad ran the subprocess; it must only prepare the command")
	}
	got := strings.Join(next.inv.Argv, " ")
	for _, want := range []string{"onnx2oracle", "load", "all-MiniLM-L6-v2", "--name", "ALL_MINILM_L6_V2", "--task", "embedding", "--force"} {
		if !strings.Contains(got, want) {
			t.Errorf("argv %q is missing %q", got, want)
		}
	}
	// The whole point of the env route: argv stays safe to print.
	if strings.Contains(got, "topsecret") {
		t.Errorf("argv leaks the password: %q", got)
	}
	if cmd == nil {
		t.Fatal("StartLoad returned no command")
	}
	if next.stage != stageRunning {
		t.Errorf("stage = %v, want stageRunning", next.stage)
	}
}

// A preset that is not in the registry must be refused, not sent as a guess.
func TestStartLoadRefusesUnknownPreset(t *testing.T) {
	m := newModel(&fakeExec{})
	m.ans.Host = "localhost"
	m.ans.Port = "1521"
	m.ans.Service = "FREEPDB1"
	m.ans.User = "system"
	m.ans.Password = "pw"
	m.ans.Preset = ""
	if _, _, err := m.StartLoad(); err == nil {
		t.Fatal("StartLoad accepted an empty preset")
	}
}

func TestDSNValidationRejectsBadPort(t *testing.T) {
	m := newModel(&fakeExec{})
	m.ans.User = "system"
	m.ans.Password = "pw"
	m.ans.Host = "localhost"
	m.ans.Service = "FREEPDB1"
	for _, port := range []string{"0", "70000", "not-a-number"} {
		m.ans.Port = port
		if _, err := m.dsn(); err == nil {
			t.Errorf("dsn() accepted port %q", port)
		}
	}
	m.ans.Port = "1521"
	if _, err := m.dsn(); err != nil {
		t.Errorf("dsn() rejected a valid port: %v", err)
	}
}

// The offline/error path is the only path testable here, so it must be explicit.
func TestSubprocessFailureSurfacesAsErrorState(t *testing.T) {
	fe := &fakeExec{out: "DPY-6005: cannot connect", err: errors.New("DPY-6005: cannot connect")}
	m := newModel(fe)
	m.ans.User = "system"
	m.ans.Password = "pw"
	m.ans.Host = "localhost"
	m.ans.Port = "1521"
	m.ans.Service = "FREEPDB1"
	m.ans.Preset = "all-MiniLM-L6-v2"
	m.ans.Confirmed = true

	next, cmd, err := m.StartLoad()
	if err != nil {
		t.Fatalf("StartLoad: %v", err)
	}
	updated, _ := next.Update(cmd())
	final, ok := updated.(Model)
	if !ok {
		t.Fatalf("Update returned %T, want Model", updated)
	}
	if final.stage != stageDone {
		t.Errorf("stage = %v, want stageDone", final.stage)
	}
	if final.err == nil {
		t.Fatal("err is nil, want the failure recorded")
	}
	if got := final.View().Content; !strings.Contains(got, "failed") {
		t.Errorf("view = %q, want it to show the failure", got)
	}
}

func TestOutputSuccessClearsError(t *testing.T) {
	m := newModel(&fakeExec{})
	updated, _ := m.Update(outputMsg{out: "OK: registered ALL_MINILM_L6_V2"})
	final := updated.(Model)
	if final.err != nil {
		t.Errorf("err = %v, want nil on success", final.err)
	}
	if !strings.Contains(final.View().Content, "OK: registered") {
		t.Errorf("view = %q, want the output", final.View().Content)
	}
}

// Drain must terminate even when the command stream is an endless blink loop.
// huh re-arms blink on every update, so this is the exact shape that hung a
// sibling repository's suite for minutes.
func TestDrainTerminatesOnBlinkLoop(t *testing.T) {
	m := newModel(&fakeExec{})
	blink := func() tea.Msg { return cursor.BlinkMsg{} }
	done := make(chan tea.Model, 1)
	go func() { done <- Drain(m, blink, 0) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Drain did not terminate on a cursor.BlinkMsg stream")
	}
}

func TestWindowSizeKeepsViewportUsable(t *testing.T) {
	m := newModel(&fakeExec{})
	updated, _ := m.Update(tea.WindowSizeMsg{Width: 40, Height: 10})
	final := updated.(Model)
	if final.viewport.Width() <= 0 || final.viewport.Height() <= 0 {
		t.Fatalf("viewport = %dx%d, want a positive size", final.viewport.Width(), final.viewport.Height())
	}
}

func TestViewRendersTheFirstForm(t *testing.T) {
	m := newModel(&fakeExec{})
	// A huh form renders no fields until Init has been called, so the view is
	// asserted after the same startup the runtime performs.
	started, ok := Drain(m, m.Init(), 0).(Model)
	if !ok {
		t.Fatal("Drain did not return a Model")
	}
	if got := started.View().Content; !strings.Contains(got, "Oracle user") {
		t.Errorf("view = %q, want the connection form", got)
	}
}

// typeRunes sends each rune as its own key press, the way a terminal does.
func typeRunes(t *testing.T, m tea.Model, s string) tea.Model {
	t.Helper()
	for _, r := range s {
		updated, cmd := m.Update(tea.KeyPressMsg{Code: r, Text: string(r)})
		m = Drain(updated, cmd, 0)
	}
	return m
}

func pressEnter(t *testing.T, m tea.Model) tea.Model {
	t.Helper()
	updated, cmd := m.Update(tea.KeyPressMsg{Code: tea.KeyEnter})
	return Drain(updated, cmd, 0)
}

// The credentials page must render its fields, and the target page must be
// reachable: a form that clips a field, or that cannot be advanced past, is a
// form that cannot be answered.
func TestFormAsksForEveryConnectionField(t *testing.T) {
	m := newModel(&fakeExec{})
	cur := tea.Model(m)
	cur = Drain(cur, cur.Init(), 0)

	first := cur.(Model).View().Content
	for _, want := range []string{"Oracle user", "Password"} {
		if !strings.Contains(first, want) {
			t.Errorf("first page is missing the %q field", want)
		}
	}

	// The user field is pre-filled from the seed, so Enter validates and moves on;
	// the password field is deliberately not pre-filled, so it has to be typed.
	cur = pressEnter(t, cur)
	cur = typeRunes(t, cur, "pw")
	cur = pressEnter(t, cur)

	final, ok := cur.(Model)
	if !ok {
		t.Fatalf("Update returned %T, want Model", cur)
	}
	target := final.View().Content
	for _, want := range []string{"Host", "Port", "Service"} {
		if !strings.Contains(target, want) {
			t.Errorf("target page is missing the %q field", want)
		}
	}
}

// A blank password must stop the form rather than let it advance on an empty
// credential: the validator is the only thing standing between the user and a
// confusing connection failure.
func TestBlankPasswordBlocksTheForm(t *testing.T) {
	m := newModel(&fakeExec{})
	cur := tea.Model(m)
	cur = Drain(cur, cur.Init(), 0)
	cur = pressEnter(t, cur) // move to the password field
	cur = pressEnter(t, cur) // try to advance with it blank

	got := cur.(Model).View().Content
	if strings.Contains(got, "Host") {
		t.Error("the form advanced to the target page with a blank password")
	}
}

func TestSelectedPresetCarriesTheOracleName(t *testing.T) {
	m := newModel(&fakeExec{})
	m.ans.Preset = "ms-marco-MiniLM-L-6-v2"
	got := m.selectedPreset()
	if got.OracleName != "MS_MARCO_MINILM_L_6_V2" || got.Task != "reranker" {
		t.Fatalf("selectedPreset = %+v, want the reranker row", got)
	}
}
