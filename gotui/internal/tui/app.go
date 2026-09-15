// Package tui is the huh-driven front-end for onnx2oracle.
//
// It gathers connection details, picks a curated preset and confirms exactly
// what will be registered before shelling out to the real Python CLI. All
// answers live in one heap-allocated struct: binding a huh field to a field of a
// value-typed Bubble Tea model silently persists defaults, because Update
// receives a copy.
package tui

import (
	"fmt"
	"strings"
	"time"

	"charm.land/bubbles/v2/cursor"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"

	"github.com/jasperan/onnx2oracle/gotui/internal/dsn"
	"github.com/jasperan/onnx2oracle/gotui/internal/huhstyle"
	"github.com/jasperan/onnx2oracle/gotui/internal/presets"
	"github.com/jasperan/onnx2oracle/gotui/internal/run"
)

// Executor runs a prepared invocation. It is an interface so the form logic can
// be tested without a database, which is the only way it can be tested here.
type Executor interface {
	Run(inv run.Invocation, timeout time.Duration) (string, error)
}

type stage int

const (
	stageConnect stage = iota
	stagePreset
	stageConfirm
	stageRunning
	stageDone
)

// answers is shared by every form. It is a pointer so huh writes through to it.
type answers struct {
	User     string
	Password string
	Host     string
	Port     string
	Service  string

	Preset    string
	Force     bool
	Confirmed bool
}

// outputMsg carries the subprocess result back into the update loop.
type outputMsg struct {
	out string
	err error
}

// Model is the Bubble Tea model for the whole flow.
type Model struct {
	stage   stage
	ans     *answers
	form    *huh.Form
	presets []presets.Preset
	exec    Executor
	binary  string
	timeout time.Duration

	viewport viewport.Model
	inv      run.Invocation
	out      string
	err      error

	width  int
	height int
}

// Options configures a Model.
type Options struct {
	Presets []presets.Preset
	Exec    Executor
	Binary  string
	Timeout time.Duration
	// Seed preselects connection values, e.g. from flags or the environment.
	Seed dsn.DSN
}

// New builds the model and the first form.
func New(opts Options) Model {
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}
	ans := &answers{
		User:    opts.Seed.User,
		Host:    opts.Seed.Host,
		Service: opts.Seed.Service,
	}
	if opts.Seed.Port > 0 {
		ans.Port = fmt.Sprintf("%d", opts.Seed.Port)
	}
	m := Model{
		stage:    stageConnect,
		ans:      ans,
		presets:  opts.Presets,
		exec:     opts.Exec,
		binary:   opts.Binary,
		timeout:  timeout,
		viewport: viewport.New(viewport.WithWidth(80), viewport.WithHeight(20)),
		width:    80,
		height:   24,
	}
	m.form = m.sized(m.connectForm())
	return m
}

// sized pins a form to the model's current dimensions. Without this a form keeps
// huh's zero default and clips fields off the bottom of the group.
func (m *Model) sized(f *huh.Form) *huh.Form { return f.WithWidth(m.width).WithHeight(m.height) }

func (m *Model) dsn() (dsn.DSN, error) {
	port := 0
	if m.ans.Port != "" {
		if _, err := fmt.Sscanf(m.ans.Port, "%d", &port); err != nil {
			return dsn.DSN{}, fmt.Errorf("port %q is not a number", m.ans.Port)
		}
	}
	d := dsn.DSN{User: m.ans.User, Password: m.ans.Password, Host: m.ans.Host, Port: port, Service: m.ans.Service}
	return d, d.Validate()
}

// themed attaches the project theme, accessibility flag and keymap. The keymap
// matters: a bare field has none, so keystrokes would be silently ignored.
func themed(f *huh.Form) *huh.Form {
	return f.
		WithTheme(huh.ThemeFunc(huhstyle.Theme)).
		WithAccessible(huhstyle.Accessible()).
		WithKeyMap(huh.NewDefaultKeyMap())
}

// ValidateDefaulted accepts an empty answer as "keep the value already in the field".
//
// It exists for accessible mode. huh's screen-reader path runs a field's validator on the
// raw line and only afterwards substitutes the field's default
// (internal/accessibility/accessibility.go:PromptString returns
// cmp.Or(strings.TrimSpace(input), defaultValue)), and it never prints that default. A
// pre-filled field whose validator rejects "" therefore keeps re-prompting on a bare
// Enter, so a screen-reader user cannot accept a value they cannot see.
func ValidateDefaulted(inner func(string) error) func(string) error {
	return func(s string) error {
		if strings.TrimSpace(s) == "" {
			return nil
		}
		return inner(s)
	}
}

// ValidateDefaultedValue is ValidateDefaulted for a field whose pre-filled value may
// itself be empty, which is the normal case here because the seed comes from flags or the
// environment. Blank stays invalid when there is nothing to keep.
func ValidateDefaultedValue(prefilled string, inner func(string) error) func(string) error {
	if strings.TrimSpace(prefilled) == "" {
		return inner
	}
	return ValidateDefaulted(inner)
}

func (m *Model) connectForm() *huh.Form {
	// Two groups, not one: five bordered fields do not fit a 24-row terminal, and
	// huh clips whatever overflows. Groups are pages, so this also gives the form a
	// natural credentials-then-target shape.
	return themed(huh.NewForm(
		huh.NewGroup(
			huh.NewInput().
				Title("Oracle user").
				Description("connection.py default: system").
				Placeholder(dsn.DefaultUser).
				Value(&m.ans.User).
				Validate(ValidateDefaultedValue(m.ans.User, huh.ValidateNotEmpty())),
			huh.NewInput().
				Title("Password").
				Description("kept in the environment, never on the command line").
				EchoMode(huh.EchoModePassword).
				Value(&m.ans.Password).
				// NOT wrapped in ValidateDefaultedValue: the password is never seeded, so
				// there is no default to keep and a blank answer must still be rejected.
				Validate(huh.ValidateNotEmpty()),
		).Title("Oracle credentials").Description("password is passed via ORACLE_DSN, never argv"),
		huh.NewGroup(
			huh.NewInput().
				Title("Host").
				Placeholder(dsn.DefaultHost).
				Value(&m.ans.Host).
				Validate(func(s string) error {
					if strings.TrimSpace(s) == "" && strings.TrimSpace(m.ans.Host) == "" {
						m.ans.Host = dsn.DefaultHost
					}
					return nil
				}),
			huh.NewInput().
				Title("Port").
				Description("1-65535").
				Placeholder(fmt.Sprintf("%d", dsn.DefaultPort)).
				Value(&m.ans.Port).
				Validate(func(s string) error {
					if strings.TrimSpace(s) == "" {
						m.ans.Port = fmt.Sprintf("%d", dsn.DefaultPort)
						return nil
					}
					var p int
					if _, err := fmt.Sscanf(s, "%d", &p); err != nil {
						return fmt.Errorf("port must be a number, got %q", s)
					}
					if p < 1 || p > 65535 {
						return fmt.Errorf("port must be between 1 and 65535, got %d", p)
					}
					return nil
				}),
			huh.NewInput().
				Title("Service").
				Placeholder(dsn.DefaultService).
				Value(&m.ans.Service).
				Validate(func(s string) error {
					if strings.TrimSpace(s) == "" && strings.TrimSpace(m.ans.Service) == "" {
						m.ans.Service = dsn.DefaultService
					}
					return nil
				}),
		).Title("Easy Connect target"),
	))
}

func (m *Model) presetForm() *huh.Form {
	opts := make([]huh.Option[string], 0, len(m.presets))
	for _, p := range m.presets {
		// Static options only: OptionsFunc yields an empty list in accessible mode.
		opts = append(opts, huh.NewOption(p.Label(), p.Name))
	}
	if m.ans.Preset == "" && len(m.presets) > 0 {
		m.ans.Preset = m.presets[0].Name
	}
	return themed(huh.NewForm(
		huh.NewGroup(
			huh.NewSelect[string]().
				Title("Model preset").
				Description("curated entries from `onnx2oracle presets`").
				Options(opts...).
				Value(&m.ans.Preset).
				Validate(func(s string) error {
					if s == "" {
						return fmt.Errorf("pick a preset")
					}
					return nil
				}),
			huh.NewConfirm().
				Title("Replace an existing registration?").
				Description("passes --force; without it a registered model is left alone").
				Affirmative("Yes, replace").
				Negative("No, keep existing").
				Value(&m.ans.Force),
		).Title("Model").Description("registered under the preset's Oracle model name"),
	))
}

func (m *Model) confirmForm() *huh.Form {
	d, err := m.dsn()
	summary := "connection unresolved"
	if err == nil {
		summary = d.Display()
	}
	p := m.selectedPreset()
	body := fmt.Sprintf("connection  %s\npreset      %s\noracle name %s\nreplace     %v",
		summary, p.Name, p.OracleName, m.ans.Force)

	return themed(huh.NewForm(
		huh.NewGroup(
			huh.NewNote().Title("Ready to load").Description(body),
			huh.NewConfirm().
				Title("Run `onnx2oracle load` now?").
				Affirmative("Load it").
				Negative("Cancel").
				Value(&m.ans.Confirmed),
		).Title("Confirm"),
	))
}

func (m Model) selectedPreset() presets.Preset {
	for _, p := range m.presets {
		if p.Name == m.ans.Preset {
			return p
		}
	}
	return presets.Preset{Name: m.ans.Preset}
}

// Init starts the first form.
func (m Model) Init() tea.Cmd { return m.form.Init() }

// Update drives the form chain and, once confirmed, the subprocess.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.viewport.SetWidth(max(20, msg.Width-4))
		m.viewport.SetHeight(max(5, msg.Height-8))
		// The form must be resized too, or a narrow terminal wraps its borders.
		m.form = m.form.WithWidth(msg.Width).WithHeight(msg.Height)
		return m, nil
	case outputMsg:
		m.stage = stageDone
		m.out = msg.out
		m.err = msg.err
		m.viewport.SetContent(msg.out)
		m.viewport.GotoTop()
		return m, nil
	case tea.KeyPressMsg:
		if msg.String() == "ctrl+c" {
			return m, tea.Quit
		}
	}

	if m.stage == stageRunning || m.stage == stageDone {
		var cmd tea.Cmd
		m.viewport, cmd = m.viewport.Update(msg)
		return m, cmd
	}

	form, cmd := m.form.Update(msg)
	if f, ok := form.(*huh.Form); ok {
		m.form = f
	}
	if m.form.State == huh.StateAborted {
		return m, tea.Quit
	}
	if m.form.State != huh.StateCompleted {
		return m, cmd
	}

	switch m.stage {
	case stageConnect:
		if _, err := m.dsn(); err != nil {
			m.stage = stageDone
			m.err = err
			m.out = "connection invalid: " + err.Error()
			return m, nil
		}
		m.stage = stagePreset
		m.form = m.sized(m.presetForm())
		return m, tea.Batch(cmd, m.form.Init())
	case stagePreset:
		m.stage = stageConfirm
		m.form = m.sized(m.confirmForm())
		return m, tea.Batch(cmd, m.form.Init())
	case stageConfirm:
		if !m.ans.Confirmed {
			m.stage = stageDone
			m.out = "cancelled; nothing was loaded."
			return m, nil
		}
		next, cmd2, err := m.StartLoad()
		if err != nil {
			m.stage = stageDone
			m.err = err
			m.out = "connection invalid: " + err.Error()
			return m, nil
		}
		m = next
		return m, tea.Batch(cmd, cmd2)
	}
	return m, cmd
}

// StartLoad turns the gathered answers into a prepared invocation and the
// command that executes it.
//
// Exported as a seam: the huh forms cannot be driven headlessly, so the decision
// this makes -- which preset, which flags, which environment -- is tested
// directly instead of through keystrokes.
func (m Model) StartLoad() (Model, tea.Cmd, error) {
	d, err := m.dsn()
	if err != nil {
		return m, nil, err
	}
	p := m.selectedPreset()
	if p.Name == "" {
		return m, nil, fmt.Errorf("no preset selected")
	}
	m.inv = run.Load(d, m.binary, run.LoadRequest{
		Preset: p.Name,
		Name:   p.OracleName,
		Task:   p.Task,
		Force:  m.ans.Force,
	})
	m.stage = stageRunning
	m.out = "$ " + m.inv.CommandLine() + "\n\nrunning…"
	m.viewport.SetContent(m.out)
	exec, inv, timeout := m.exec, m.inv, m.timeout
	return m, func() tea.Msg {
		out, err := exec.Run(inv, timeout)
		if err != nil {
			// An unreachable Oracle is the expected offline outcome and must
			// surface as a clear error, never as a hang.
			return outputMsg{out: out + "\n\n" + err.Error(), err: err}
		}
		return outputMsg{out: out}
	}, nil
}

// View renders the active stage.
func (m Model) View() tea.View {
	var b strings.Builder
	switch m.stage {
	case stageConnect, stagePreset, stageConfirm:
		b.WriteString(m.form.View())
	case stageRunning, stageDone:
		b.WriteString(titleStyle.Render("onnx2oracle — load") + "\n")
		if m.err != nil {
			b.WriteString(errorStyle.Render("failed: "+m.err.Error()) + "\n")
		}
		b.WriteString(m.viewport.View() + "\n")
		b.WriteString(hintStyle.Render("q quits"))
	}
	return tea.NewView(b.String())
}

var (
	titleStyle = lipgloss.NewStyle().Bold(true)
	errorStyle = lipgloss.NewStyle().Bold(true)
	hintStyle  = lipgloss.NewStyle().Faint(true)
)

// Drain runs a command tree to completion, discarding cursor blink ticks.
//
// huh re-arms the text input's blink on every update and each tick sleeps about
// half a second; feeding a BlinkMsg back into Update re-arms it forever, so an
// undiscriminating drain never terminates. Exported because the tests use it.
func Drain(m tea.Model, cmd tea.Cmd, depth int) tea.Model {
	if cmd == nil || depth > 64 {
		return m
	}
	msg := cmd()
	if msg == nil {
		return m
	}
	if batch, ok := msg.(tea.BatchMsg); ok {
		for _, c := range batch {
			m = Drain(m, c, depth+1)
		}
		return m
	}
	if _, blink := msg.(cursor.BlinkMsg); blink {
		return m
	}
	next, nextCmd := m.Update(msg)
	return Drain(next, nextCmd, depth+1)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
