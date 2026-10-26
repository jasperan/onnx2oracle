package tui

import (
	"errors"
	"testing"
)

func TestFailureLine(t *testing.T) {
	exit := errors.New("exit status 1")
	cases := []struct {
		name, out, want string
	}{
		{
			name: "traceback header wins over the code frame",
			out: "Traceback (most recent call last):\n  File \"x.py\", line 3, in f\n    raise TypeError(m)\n" +
				"oracledb.exceptions.DatabaseError: ORA-01017: invalid credential\n",
			want: "oracledb.exceptions.DatabaseError: ORA-01017: invalid credential",
		},
		{
			name: "no traceback: the last line is the failure, not the first progress line",
			out:  "Downloading model...\nBuilding graph...\nConnection refused: localhost:1521\n",
			want: "Connection refused: localhost:1521",
		},
		{
			name: "box borders and ANSI are skipped",
			out:  "starting\n\x1b[31m│ Cannot reach the database │\x1b[0m\n╰───────╯\n",
			want: "Cannot reach the database",
		},
		{name: "no output falls back to the exit error", out: "  \n", want: "exit status 1"},
	}
	for _, tc := range cases {
		if got := failureLine(tc.out, exit); got != tc.want {
			t.Errorf("%s: failureLine = %q, want %q", tc.name, got, tc.want)
		}
	}
}
