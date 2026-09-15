// Command onnx2oracle-tui is a Go front-end for the onnx2oracle Python CLI.
//
// It is an alternative way to run the same engine, not a replacement: every
// action shells out to the installed `onnx2oracle` binary over the interface
// that binary already exposes.
//
// Two paths:
//   - a terminal: a huh-driven wizard (connection, preset, confirm, then output)
//   - no terminal: flags and environment only, never a prompt
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/jasperan/onnx2oracle/gotui/internal/cli"
	"github.com/jasperan/onnx2oracle/gotui/internal/dsn"
	"github.com/jasperan/onnx2oracle/gotui/internal/huhstyle"
	"github.com/jasperan/onnx2oracle/gotui/internal/presets"
	"github.com/jasperan/onnx2oracle/gotui/internal/run"
	"github.com/jasperan/onnx2oracle/gotui/internal/tui"
)

const version = "0.1.0"

func main() {
	os.Exit(realMain())
}

func realMain() int {
	var (
		opts    cli.Options
		binary  = flag.String("bin", "onnx2oracle", "path to the onnx2oracle CLI")
		timeout = flag.Duration("timeout", tui.DefaultTimeout, "subprocess timeout")
		list    = flag.Bool("list-presets", false, "print the available presets and exit")
		ver     = flag.Bool("version", false, "print the front-end version and exit")
	)
	flag.StringVar(&opts.User, "user", "", "Oracle user")
	flag.StringVar(&opts.Password, "password", "", "Oracle password (env is preferred; argv is visible in ps)")
	flag.StringVar(&opts.Host, "host", "", "Oracle host")
	flag.StringVar(&opts.Port, "port", "", "Oracle port")
	flag.StringVar(&opts.Service, "service", "", "Oracle service")
	flag.StringVar(&opts.DSN, "dsn", "", "full DSN: user/password@host:port/service")
	flag.StringVar(&opts.Preset, "preset", "", "preset to load (see -list-presets)")
	flag.StringVar(&opts.Name, "model", "", "override the Oracle model name")
	flag.BoolVar(&opts.Load, "load", false, "perform the registration")
	flag.BoolVar(&opts.Yes, "yes", false, "implied by -load; accepted for scripts")
	flag.Parse()

	if *ver {
		fmt.Printf("onnx2oracle-tui %s\n", version)
		return 0
	}

	if *list {
		got, err := presets.Load(*binary, 30*time.Second)
		if err != nil {
			fmt.Fprintf(os.Stderr, "could not list presets: %v\n", err)
			return 1
		}
		for _, p := range got {
			fmt.Println(p.Label())
		}
		return 0
	}

	if !huhstyle.Interactive() {
		return runNonInteractive(opts, *binary, *timeout)
	}

	got, err := presets.Load(*binary, 60*time.Second)
	if err != nil {
		// The wizard is unusable without the registry: every option would be
		// invented. Report why rather than showing an empty list.
		fmt.Fprintf(os.Stderr, "could not read the preset registry: %v\n", err)
		fmt.Fprintln(os.Stderr, "Fix the onnx2oracle install, or use the flags: -user -host -port -service -preset -load")
		return 1
	}

	seed := seedFromOptions(opts)
	m := tui.New(tui.Options{
		Presets: got,
		Exec:    tui.ShellExec{},
		Binary:  *binary,
		Timeout: *timeout,
		Seed:    seed,
	})
	if _, err := tea.NewProgram(m).Run(); err != nil {
		fmt.Fprintf(os.Stderr, "tui failed: %v\n", err)
		return 1
	}
	return 0
}

// seedFromOptions pre-fills the wizard from the flags so a user who already
// knows their target does not retype it. It is a convenience only: the wizard
// still validates everything.
func seedFromOptions(o cli.Options) dsn.DSN {
	if o.DSN != "" {
		if d, err := dsn.Parse(o.DSN); err == nil {
			return d
		}
	}
	d := dsn.Local()
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
		if p, err := strconv.Atoi(o.Port); err == nil {
			d.Port = p
		}
	}
	return d
}

// runNonInteractive resolves flags and environment with no terminal at all.
func runNonInteractive(opts cli.Options, binary string, timeout time.Duration) int {
	plan, err := cli.Resolve(opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		return 2
	}
	fmt.Printf("connection: %s\npreset:     %s\n", plan.DSN.Display(), plan.Preset)
	if !opts.ShouldLoad() {
		fmt.Println("dry run: pass -load (or -yes) to register it")
		return 0
	}
	inv := run.Load(plan.DSN, binary, run.LoadRequest{Preset: plan.Preset, Name: plan.Name})
	out, err := tui.ShellExec{}.Run(inv, timeout)
	if out != "" {
		fmt.Print(out)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "load failed: %v\n", err)
		return 1
	}
	return 0
}
