// Package dsn models an Oracle connection the way the onnx2oracle CLI does.
//
// The field set, the local defaults and the Easy Connect shape ("host:port/service")
// are taken from src/onnx2oracle/connection.py so that the Go front-end resolves a
// connection identically to the Python CLI it drives.
package dsn

import (
	"errors"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// Defaults mirrored from connection.py (_local_dsn).
const (
	DefaultUser     = "system"
	DefaultPassword = "onnx2oracle"
	DefaultHost     = "localhost"
	DefaultPort     = 1521
	DefaultService  = "FREEPDB1"
)

// Env vars honoured by connection.py. ORACLE_DSN carries a full DSN and
// ORACLE_PORT/ORACLE_PWD supply the local-target pieces.
const (
	EnvDSN  = "ORACLE_DSN"
	EnvPort = "ORACLE_PORT"
	EnvPwd  = "ORACLE_PWD"
)

// DSN is a resolved Oracle connection. ConnectString holds a full Easy Connect
// descriptor when the source was not the user/password@host:port/service form.
type DSN struct {
	User          string
	Password      string
	Host          string
	Port          int
	Service       string
	ConnectString string
}

// Local returns the zero-config local target, honouring ORACLE_PORT/ORACLE_PWD
// exactly like connection.py's _local_dsn(), including its failure: a set but
// non-integer ORACLE_PORT is an error rather than a silent fallback to 1521.
func Local() (DSN, error) {
	port := DefaultPort
	if raw, ok := os.LookupEnv(EnvPort); ok {
		n, err := strconv.Atoi(raw)
		if err != nil {
			return DSN{}, fmt.Errorf("%s must be an integer, got %q", EnvPort, raw)
		}
		port = n
	}
	password := DefaultPassword
	if pw := os.Getenv(EnvPwd); pw != "" {
		password = pw
	}
	return DSN{User: DefaultUser, Password: password, Host: DefaultHost, Port: port, Service: DefaultService}, nil
}

var easyConnect = regexp.MustCompile(`^(?P<host>[^:/]+):(?P<port>\d+)/(?P<service>.+)$`)

// Parse accepts "user/password@host:port/service". The password may contain '@',
// so the split scans from the right, matching DSN.parse in connection.py. Errors
// never quote the input: it carries the password.
func Parse(raw string) (DSN, error) {
	slash := strings.Index(raw, "/")
	if slash <= 0 {
		return DSN{}, errors.New("malformed DSN: expected user/password@host:port/service")
	}
	user := raw[:slash]
	rest := raw[slash+1:]

	idx := strings.LastIndex(rest, "@")
	for idx >= 0 {
		if m := easyConnect.FindStringSubmatch(rest[idx+1:]); m != nil {
			port, err := strconv.Atoi(m[2])
			if err != nil {
				return DSN{}, errors.New("malformed DSN: bad port")
			}
			return DSN{User: user, Password: rest[:idx], Host: m[1], Port: port, Service: m[3]}, nil
		}
		idx = strings.LastIndex(rest[:idx], "@")
	}

	// No host:port/service tail: treat everything after the last '@' as an opaque
	// connect descriptor, the way connection.py does.
	if idx = strings.LastIndex(rest, "@"); idx >= 0 && idx < len(rest)-1 {
		cs := rest[idx+1:]
		return DSN{User: user, Password: rest[:idx], Host: cs, ConnectString: cs}, nil
	}
	return DSN{}, errors.New("malformed DSN: no '@' target")
}

// Validate reports the first field that cannot produce a usable connection.
// Port is only checked for a non-descriptor connection, because a connect
// descriptor carries its own topology.
func (d DSN) Validate() error {
	if strings.TrimSpace(d.User) == "" {
		return errors.New("user must not be empty")
	}
	if strings.TrimSpace(d.Password) == "" {
		return errors.New("password must not be empty")
	}
	if d.ConnectString != "" {
		return nil
	}
	if strings.TrimSpace(d.Host) == "" {
		return errors.New("host must not be empty")
	}
	if d.Port < 1 || d.Port > 65535 {
		return fmt.Errorf("port must be between 1 and 65535, got %d", d.Port)
	}
	if strings.TrimSpace(d.Service) == "" {
		return errors.New("service must not be empty")
	}
	return nil
}

// OracleDSN returns the string for oracledb's connect(dsn=...): either the
// opaque descriptor or "host:port/service".
func (d DSN) OracleDSN() string {
	if d.ConnectString != "" {
		return d.ConnectString
	}
	return fmt.Sprintf("%s:%d/%s", d.Host, d.Port, d.Service)
}

// String is the password-free form, so a DSN passed to a formatter or logger
// cannot leak credentials. Use ConnectionString for the full credentialed form.
func (d DSN) String() string { return d.Display() }

// ConnectionString is the full user/password@host:port/service form the Python
// CLI accepts in ORACLE_DSN. It contains the password and round-trips through
// Parse; never log it.
func (d DSN) ConnectionString() string {
	if d.ConnectString != "" {
		return fmt.Sprintf("%s/%s@%s", d.User, d.Password, d.ConnectString)
	}
	return fmt.Sprintf("%s/%s@%s:%d/%s", d.User, d.Password, d.Host, d.Port, d.Service)
}

// Display is the safe-for-logs representation (no password), matching
// DSN.display in connection.py.
func (d DSN) Display() string {
	if d.ConnectString != "" {
		target := d.ConnectString
		if strings.HasPrefix(strings.TrimSpace(target), "(") {
			target = "<connect-descriptor>"
		}
		return fmt.Sprintf("%s@%s", d.User, target)
	}
	return fmt.Sprintf("%s@%s:%d/%s", d.User, d.Host, d.Port, d.Service)
}
