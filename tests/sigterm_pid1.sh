#!/bin/bash
# sigterm_pid1.sh <unit> <port> <binary> <model_dir> [server args...]
# Starts the server as PID 1 of a new PID namespace (unshare -rpf, as in a container, where a signal
# without a handler is ignored) inside a transient systemd user unit, waits for /health, sends one
# SIGTERM from the parent namespace and measures the time until the process is gone.
# Prints "SIGTERM_EXIT_MS=<n> rc=<exit status>"; exits 1 when the server took 2 s or more (or never
# exited: it is then SIGKILLed after 5 s). MUSL_LD: a musl loader to run a musl binary on a glibc host,
# e.g. MUSL_LD="<toolchain>/lib/ld-musl-x86_64.so.1 --library-path <toolchain>/lib".
set -u
U=$1; PORT=$2; BIN=$3; MD=$4; shift 4
systemd-run --user --quiet --unit="$U" -p KillSignal=SIGKILL -p TimeoutStopSec=1 \
  unshare -rpf ${MUSL_LD:-} "$BIN" --model-path "$MD" --port "$PORT" "$@"
for i in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 0.1
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "NO_HEALTH"; systemctl --user stop "$U"; exit 2; }
PID=$(pgrep -P "$(systemctl --user show -p MainPID --value "$U")")
[ -n "$PID" ] || { echo "NO_CHILD"; systemctl --user stop "$U"; exit 2; }
echo "server pid $PID: $(grep NSpid /proc/$PID/status)"
t0=$(date +%s%N)
kill -TERM "$PID"
ms=-1
for i in $(seq 1 1000); do
  if ! kill -0 "$PID" 2>/dev/null; then ms=$(( ($(date +%s%N) - t0) / 1000000 )); break; fi
  sleep 0.005
done
[ "$ms" = -1 ] && ms=">5000"
rc=$(systemctl --user show -p ExecMainStatus --value "$U" 2>/dev/null)
systemctl --user stop "$U" 2>/dev/null; systemctl --user reset-failed "$U" 2>/dev/null
echo "SIGTERM_EXIT_MS=$ms rc=$rc"
[ "$ms" != ">5000" ] && [ "$ms" -lt 2000 ]
