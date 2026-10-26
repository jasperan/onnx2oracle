package dsn

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

// mustLocal returns the local target with a clean environment, so tests do not
// depend on the developer's ORACLE_PORT / ORACLE_PWD.
func mustLocal(t *testing.T) DSN {
	t.Helper()
	t.Setenv(EnvPort, "1521") // registers restoration of the original value
	os.Unsetenv(EnvPort)
	t.Setenv(EnvPwd, "")
	got, err := Local()
	if err != nil {
		t.Fatalf("Local: %v", err)
	}
	return got
}

func TestLocalDefaults(t *testing.T) {
	got := mustLocal(t)
	if got.User != "system" || got.Host != "localhost" || got.Port != 1521 || got.Service != "FREEPDB1" {
		t.Fatalf("Local() = %+v, want the connection.py local defaults", got)
	}
	if got.Password != DefaultPassword {
		t.Errorf("password = %q, want %q", got.Password, DefaultPassword)
	}
}

func TestLocalHonoursEnv(t *testing.T) {
	t.Setenv(EnvPort, "1600")
	t.Setenv(EnvPwd, "s3cret")
	got, err := Local()
	if err != nil {
		t.Fatalf("Local: %v", err)
	}
	if got.Port != 1600 || got.Password != "s3cret" {
		t.Fatalf("Local() = %+v, want port 1600 and the env password", got)
	}
}

// connection.py raises on a non-integer ORACLE_PORT; falling back to 1521 would
// silently connect to the wrong database.
func TestLocalRejectsInvalidPort(t *testing.T) {
	for _, raw := range []string{"abc", "15 21", "", "1521.5"} {
		t.Setenv(EnvPort, raw)
		if _, err := Local(); err == nil {
			t.Errorf("Local() with %s=%q = nil error, want a rejection", EnvPort, raw)
		}
	}
}

func TestParseEasyConnect(t *testing.T) {
	got, err := Parse("system/hunter2@db.example.com:1522/MYPDB1")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	want := DSN{User: "system", Password: "hunter2", Host: "db.example.com", Port: 1522, Service: "MYPDB1"}
	if got != want {
		t.Fatalf("Parse = %+v, want %+v", got, want)
	}
}

// A password containing '@' must still resolve the trailing @host:port/service.
func TestParsePasswordWithAtSign(t *testing.T) {
	got, err := Parse("admin/p@ss@word@localhost:1521/FREEPDB1")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got.Password != "p@ss@word" || got.Host != "localhost" || got.Service != "FREEPDB1" {
		t.Fatalf("Parse = %+v, want the password to keep its @ characters", got)
	}
}

func TestParseConnectDescriptor(t *testing.T) {
	got, err := Parse("system/pw@mydb_high")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got.ConnectString != "mydb_high" || got.OracleDSN() != "mydb_high" {
		t.Fatalf("Parse = %+v, want an opaque connect descriptor", got)
	}
}

func TestParseRejectsMalformed(t *testing.T) {
	for _, raw := range []string{"", "nouser", "user/", "user/pw", "user/pw@"} {
		if _, err := Parse(raw); err == nil {
			t.Errorf("Parse(%q) = nil error, want a rejection", raw)
		}
	}
}

func TestValidateRejections(t *testing.T) {
	base := mustLocal(t)
	cases := map[string]func(DSN) DSN{
		"empty user":    func(d DSN) DSN { d.User = "  "; return d },
		"empty pass":    func(d DSN) DSN { d.Password = ""; return d },
		"empty host":    func(d DSN) DSN { d.Host = ""; return d },
		"empty service": func(d DSN) DSN { d.Service = ""; return d },
		"port zero":     func(d DSN) DSN { d.Port = 0; return d },
		"port negative": func(d DSN) DSN { d.Port = -1; return d },
		"port too big":  func(d DSN) DSN { d.Port = 70000; return d },
	}
	for name, mutate := range cases {
		if err := mutate(base).Validate(); err == nil {
			t.Errorf("Validate(%s) = nil, want an error", name)
		}
	}
	if err := base.Validate(); err != nil {
		t.Errorf("Validate(default local) = %v, want nil", err)
	}
}

// A connect descriptor has no host/port/service of its own, so those checks must
// stand down rather than reject a usable descriptor.
func TestValidateSkipsTopologyForDescriptor(t *testing.T) {
	d := DSN{User: "system", Password: "pw", ConnectString: "(DESCRIPTION=(ADDRESS=(HOST=x)))"}
	if err := d.Validate(); err != nil {
		t.Fatalf("Validate(descriptor) = %v, want nil", err)
	}
}

func TestOracleDSNShape(t *testing.T) {
	got := mustLocal(t)
	if want := "localhost:1521/FREEPDB1"; got.OracleDSN() != want {
		t.Errorf("OracleDSN() = %q, want %q", got.OracleDSN(), want)
	}
}

func TestConnectionStringRoundTripsThroughParse(t *testing.T) {
	for _, orig := range []DSN{
		{User: "system", Password: "pw", Host: "h", Port: 1521, Service: "FREEPDB1"},
		{User: "system", Password: "p@ss", Host: "mydb_high", ConnectString: "mydb_high"},
	} {
		back, err := Parse(orig.ConnectionString())
		if err != nil {
			t.Fatalf("Parse(ConnectionString()): %v", err)
		}
		if back != orig {
			t.Fatalf("round trip = %+v, want %+v", back, orig)
		}
	}
}

// Anything that formats a DSN with %v / %s / %+v must stay free of the password.
func TestStringAndDisplayHidePassword(t *testing.T) {
	for _, d := range []DSN{
		{User: "system", Password: "topsecret", Host: "localhost", Port: 1521, Service: "FREEPDB1"},
		{User: "system", Password: "topsecret", Host: "mydb_high", ConnectString: "mydb_high"},
	} {
		for name, got := range map[string]string{"Display": d.Display(), "String": d.String(), "%v": fmt.Sprintf("%v", d)} {
			if strings.Contains(got, "topsecret") {
				t.Errorf("%s = %q, must not leak the password", name, got)
			}
		}
	}
}

// Parse errors must not echo the raw input, which carries the password.
func TestParseErrorsDoNotEchoInput(t *testing.T) {
	for _, raw := range []string{"topsecret", "user/topsecret", "user/topsecret@", "user/topsecret@h:99999999999999999999/S"} {
		_, err := Parse(raw)
		if err == nil {
			t.Errorf("Parse(%q) = nil error, want a rejection", raw)
			continue
		}
		if strings.Contains(err.Error(), "topsecret") {
			t.Errorf("Parse error %q leaks the input", err)
		}
	}
}
