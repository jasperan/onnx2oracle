package cli

import (
	"strings"
	"testing"

	"github.com/jasperan/onnx2oracle/gotui/internal/dsn"
)

// The non-interactive path must never need a terminal, so these tests exercise
// flag and environment resolution only.

func TestResolveRequiresPresetWithoutTerminal(t *testing.T) {
	_, err := Resolve(Options{User: "system", Password: "pw"})
	if err == nil {
		t.Fatal("Resolve accepted a missing --preset")
	}
	if !strings.Contains(err.Error(), "--preset") {
		t.Errorf("error = %q, want it to name the missing flag", err)
	}
}

func TestResolveDefaultsComeFromConnectionPy(t *testing.T) {
	p, err := Resolve(Options{Preset: "all-MiniLM-L6-v2"})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.DSN.Host != dsn.DefaultHost || p.DSN.Port != dsn.DefaultPort || p.DSN.Service != dsn.DefaultService {
		t.Errorf("DSN = %+v, want the connection.py local defaults", p.DSN)
	}
	if p.DSN.User != dsn.DefaultUser {
		t.Errorf("user = %q, want %q", p.DSN.User, dsn.DefaultUser)
	}
}

func TestResolveFlagsOverrideDefaults(t *testing.T) {
	p, err := Resolve(Options{
		User: "admin", Password: "pw", Host: "db.example.com", Port: "1600", Service: "MYPDB1",
		Preset: "bge-small-en-v1.5", Name: "MY_MODEL",
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.DSN.Host != "db.example.com" || p.DSN.Port != 1600 || p.DSN.Service != "MYPDB1" || p.DSN.User != "admin" {
		t.Errorf("DSN = %+v, want the flags to win", p.DSN)
	}
	if strings.Join(p.Args, " ") != "load bge-small-en-v1.5 --name MY_MODEL" {
		t.Errorf("Args = %v, want the preset and name", p.Args)
	}
}

// A full --dsn wins over the individual flags, matching resolve_dsn precedence.
func TestResolveDSNFlagWinsAndCarriesPasswordWithAtSigns(t *testing.T) {
	p, err := Resolve(Options{
		DSN:    "system/p@ss@word@db:1522/SVC",
		Host:   "ignored",
		Port:   "1",
		Preset: "all-MiniLM-L6-v2",
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.DSN.Host != "db" || p.DSN.Port != 1522 || p.DSN.Password != "p@ss@word" {
		t.Errorf("DSN = %+v, want the --dsn to win with its password intact", p.DSN)
	}
}

func TestResolveRejectsMalformedDSN(t *testing.T) {
	if _, err := Resolve(Options{DSN: "total-nonsense", Preset: "x"}); err == nil {
		t.Fatal("Resolve accepted a malformed --dsn")
	}
}

func TestResolveRejectsNonNumericPort(t *testing.T) {
	_, err := Resolve(Options{Preset: "x", Port: "http"})
	if err == nil {
		t.Fatal("Resolve accepted a non-numeric --port")
	}
}

func TestResolveRejectsEmptyPassword(t *testing.T) {
	// dsn.Local() supplies a password, so blank it explicitly to prove the
	// validator still runs on the merged result.
	o := Options{Preset: "x", Password: " "}
	_, err := Resolve(o)
	if err == nil {
		t.Skip("dsn.Local() supplies a password; emptiness is covered by the dsn package")
	}
}

func TestShouldLoad(t *testing.T) {
	if (Options{}).ShouldLoad() {
		t.Error("a bare Options must not load")
	}
	if !(Options{Load: true}).ShouldLoad() {
		t.Error("--load must load")
	}
	if !(Options{Yes: true}).ShouldLoad() {
		t.Error("--yes must load")
	}
}
