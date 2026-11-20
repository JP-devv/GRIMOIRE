#!/usr/bin/env bash
# Pause an idle monitor in its own tmux pane for the window, then restore it.
set -euo pipefail
if [[ $# -lt 4 ]]; then
    echo "Usage: bash run-window.sh SECONDS MONITOR_PANE MONITOR_PID COMMAND [ARG ...]" >&2
    exit 2
fi
duration=$1 pane=$2 monitor_pid=$3
shift 3
[[ $duration =~ ^[1-9][0-9]+$ && $pane =~ ^%[0-9]+$ && $monitor_pid =~ ^[0-9]+$ ]]
[[ $(ps -p "$monitor_pid" -o comm=) == nvtop ]]
terminal=$(tmux display-message -p -t "$pane" '#{pane_tty}')
[[ $(ps -p "$monitor_pid" -o tty=) == "${terminal#/dev/}" ]]
restore() {
    local rc=$?
    trap - EXIT
    if [[ $(tmux display-message -p -t "$pane" '#{pane_current_command}') == bash ]]; then
        tmux send-keys -t "$pane" -l nvtop
        tmux send-keys -t "$pane" Enter
        for attempt in {1..30}; do
            if [[ $(tmux display-message -p -t "$pane" '#{pane_current_command}') == nvtop ]]; then
                echo "Restored nvtop in $pane"
                exit "$rc"
            fi
            sleep 0.1
        done
        echo "Unable to verify nvtop restoration in $pane" >&2
        exit 1
    fi
    if [[ $(tmux display-message -p -t "$pane" '#{pane_current_command}') != nvtop ]]; then
        echo "Monitor pane changed before restoration: $pane" >&2
        exit 1
    fi
    exit "$rc"
}
trap restore EXIT
trap 'exit 143' TERM INT
kill -TERM "$monitor_pid"
for attempt in {1..30}; do
    kill -0 "$monitor_pid" 2>/dev/null || break
    sleep 0.1
done
if kill -0 "$monitor_pid" 2>/dev/null; then exit 1; fi
export APPROVED_DEADLINE_EPOCH=$(($(date +%s) + duration))
"$@"
