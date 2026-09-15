package tui

import (
	"bytes"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/jasperan/onnx2oracle/gotui/internal/dsn"
)

// This file covers the accessible (screen-reader) path.
//
// Why it exists: huh's accessible prompts run a field's validator on the raw line and only
// afterwards substitute the field's default, and they never print that default. A pre-filled
// field whose validator rejects "" therefore re-prompts on every bare Enter, so a
// screen-reader user cannot accept a value they cannot see -- which for this tool means they
// cannot get past the connection form at all when a seed is supplied.
//
// Note: app_test.go says "the forms cannot be driven headlessly". That is true of the normal
// Bubble Tea path; the accessible path is reachable from a test, and these tests use it.

var errBlank = errors.New("blank answer rejected")

// runConnectFormAccessible drives the model's real connection form through huh's accessible
// path with scripted input, returning everything it wrote.
func runConnectFormAccessible(t *testing.T, m *Model, input string) string {
	t.Helper()
	var out bytes.Buffer
	f := m.connectForm().
		WithAccessible(true).
		WithInput(strings.NewReader(input)).
		WithOutput(&out)

	done := make(chan error, 1)
	go func() { done <- f.Run() }()

	select {
	case err := <-done:
		if err != nil {
			t.Logf("Run returned %v (the password field needs a tty; huh ignores this too)", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatalf("accessible form did not finish within 10s; output so far:\n%s", out.String())
	}
	return out.String()
}

// TestValidateDefaultedAcceptsBlank is the unit half of the fix.
func TestValidateDefaultedAcceptsBlank(t *testing.T) {
	inner := func(s string) error {
		if strings.TrimSpace(s) == "" {
			return errBlank
		}
		if s == "bad" {
			return errors.New("not usable")
		}
		return nil
	}
	wrapped := ValidateDefaulted(inner)

	for _, in := range []string{"", "   ", "\t"} {
		if err := wrapped(in); err != nil {
			t.Errorf("ValidateDefaulted(inner)(%q) = %v, want nil", in, err)
		}
	}
	if err := wrapped("bad"); err == nil {
		t.Error(`ValidateDefaulted(inner)("bad") = nil, want the inner validator's error`)
	}
	if err := wrapped("system"); err != nil {
		t.Errorf(`ValidateDefaulted(inner)("system") = %v, want nil`, err)
	}
}

// TestValidateDefaultedValueOnlyRelaxesWhenSomethingIsKept is the guard: the user is usually
// seeded from flags or the environment, and either may be absent.
func TestValidateDefaultedValueOnlyRelaxesWhenSomethingIsKept(t *testing.T) {
	inner := func(s string) error {
		if strings.TrimSpace(s) == "" {
			return errBlank
		}
		return nil
	}

	if err := ValidateDefaultedValue("", inner)(""); err == nil {
		t.Error("empty seed accepted a blank answer; the required user field was weakened")
	}
	if err := ValidateDefaultedValue("   ", inner)(""); err == nil {
		t.Error("whitespace-only seed accepted a blank answer")
	}
	if err := ValidateDefaultedValue("system", inner)(""); err != nil {
		t.Errorf("seeded user rejected blank: %v", err)
	}
}

// TestAccessibleBlankAnswerKeepsTheSeededUser is the regression test proper. With a seed (the
// documented --user default), a bare Enter must keep it instead of looping on
// "input cannot be empty".
func TestAccessibleBlankAnswerKeepsTheSeededUser(t *testing.T) {
	m := New(Options{
		Presets: samplePresets(),
		Exec:    &fakeExec{},
		Binary:  "onnx2oracle",
		Timeout: time.Second,
		Seed:    dsn.DSN{User: "system", Password: "topsecret", Host: "localhost", Port: 1521, Service: "FREEPDB1"},
	})
	seeded := m.ans.User
	if seeded == "" {
		t.Fatal("the user is not seeded; the test would prove nothing")
	}

	out := runConnectFormAccessible(t, &m, strings.Repeat("\n", 6))
	if strings.Contains(out, "input cannot be empty") {
		t.Errorf("a blank answer was rejected, so a screen-reader user cannot keep the seeded "+
			"user.\noutput:\n%s", out)
	}
	if m.ans.User != seeded {
		t.Errorf("user = %q after a blank answer, want the seeded %q", m.ans.User, seeded)
	}
}

// TestAccessibleStillRejectsBlankWithoutASeed is the negative half, on the real form: with no
// seed there is nothing to keep, so the required user must still refuse a blank answer.
func TestAccessibleStillRejectsBlankWithoutASeed(t *testing.T) {
	m := New(Options{
		Presets: samplePresets(),
		Exec:    &fakeExec{},
		Binary:  "onnx2oracle",
		Timeout: time.Second,
		// No Seed: the user must be typed.
	})
	if m.ans.User != "" {
		t.Fatalf("the user is seeded with %q; the test assumes it is empty", m.ans.User)
	}

	out := runConnectFormAccessible(t, &m, strings.Repeat("\n", 3))
	if !strings.Contains(out, "input cannot be empty") {
		t.Errorf("an unseeded required field accepted a blank answer, so the blank-pass "+
			"wrapper leaked beyond seeded fields.\noutput:\n%s", out)
	}
}
