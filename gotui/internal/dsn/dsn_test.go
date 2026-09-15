package dsn

import (
	"os"
	"strings"
	"testing"
)

func TestLocalDefaults(t *testing.T) {
	t.Setenv(EnvPort, "")
	t.Setenv(EnvPwd, "")
	t.Setenv(EnvPort, "1521")
	os.Unsetenv(EnvPort)

	got := Local()
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
	got := Local()
	if got.Port != 1600 || got.Password != "s3cret" {
		t.Fatalf("Local() = %+v, want port 1600 and the env password", got)
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
	base := Local()
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
	got := Local()
	if want := "localhost:1521/FREEPDB1"; got.OracleDSN() != want {
		t.Errorf("OracleDSN() = %q, want %q", got.OracleDSN(), want)
	}
}

func TestStringRoundTripsThroughParse(t *testing.T) {
	orig := DSN{User: "system", Password: "pw", Host: "h", Port: 1521, Service: "FREEPDB1"}
	back, err := Parse(orig.String())
	if err != nil {
		t.Fatalf("Parse(String()): %v", err)
	}
	if back != orig {
		t.Fatalf("round trip = %+v, want %+v", back, orig)
	}
}

func TestDisplayHidesPassword(t *testing.T) {
	d := DSN{User: "system", Password: "topsecret", Host: "localhost", Port: 1521, Service: "FREEPDB1"}
	if got := d.Display(); strings.Contains(got, "topsecret") {
		t.Errorf("Display() = %q, must not leak the password", got)
	}
}
