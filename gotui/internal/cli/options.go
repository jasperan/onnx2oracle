// Package cli resolves the non-interactive path: flags and environment decide
// everything, with no terminal involved.
//
// This is the path a cron job, a CI step or a piped shell takes, so it must
// never depend on a prompt being answerable.
package cli

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/jasperan/onnx2oracle/gotui/internal/dsn"
)

// Options are the flags the front-end accepts.
type Options struct {
	User     string
	Password string
	Host     string
	Port     string
	Service  string
	DSN      string

	Preset string
	Name   string

	// Load performs the registration. Without it the resolved plan is printed
	// and the process exits 0, so the non-interactive path is inspectable.
	Load bool
	// Yes implies Load and is accepted for scripts that expect an affirmation flag.
	Yes bool
}

// Plan is a fully resolved non-interactive decision.
type Plan struct {
	DSN    dsn.DSN
	Preset string
	Name   string
	Args   []string // human-readable echo of what will run
}

// Resolve turns flags and environment into a Plan, or reports exactly which flag
// is missing. It never reads stdin.
func Resolve(o Options) (Plan, error) {
	var d dsn.DSN
	var err error
	switch {
	case strings.TrimSpace(o.DSN) != "":
		d, err = dsn.Parse(o.DSN)
		if err != nil {
			return Plan{}, err
		}
	default:
		d = dsn.Local()
		if o.User != "" {
			d.User = o.User
		}
		if o.Password != "" {
			d.Password = o.Password
		}
		if o.Host != "" {
			d.Host = o.Host
		}
		if o.Service != "" {
			d.Service = o.Service
		}
		if o.Port != "" {
			p, convErr := strconv.Atoi(strings.TrimSpace(o.Port))
			if convErr != nil {
				return Plan{}, fmt.Errorf("--port must be a number, got %q", o.Port)
			}
			d.Port = p
		}
	}
	if err := d.Validate(); err != nil {
		return Plan{}, fmt.Errorf("connection incomplete: %w", err)
	}

	preset := strings.TrimSpace(o.Preset)
	if preset == "" {
		return Plan{}, fmt.Errorf("--preset is required when stdin is not a terminal (see `<bin> presets`)")
	}

	p := Plan{DSN: d, Preset: preset, Name: o.Name}
	p.Args = []string{"load", preset}
	if o.Name != "" {
		p.Args = append(p.Args, "--name", o.Name)
	}
	return p, nil
}

// ShouldLoad reports whether the resolved plan should be executed rather than
// merely printed.
func (o Options) ShouldLoad() bool { return o.Load || o.Yes }
