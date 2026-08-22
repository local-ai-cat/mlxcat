#!/bin/zsh
# A benchmark queue that survives the machine dying.
#
# On 2026-08-22 vllm-mlx panicked the GPU driver on the M4 Pro worker. The host
# rebooted; three passes were queued behind it as one shell chain, each polling
# the previous pass's log for a DONE marker, and all three died with the shell
# that was waiting. The markers were still on disk afterwards — nothing was
# running to read them.
#
# So the queue lives on disk instead of in a process:
#
#   bench/queue/NN-name.sh      a pass; run in numeric order
#   bench/queue/NN-name.done    it finished successfully; never re-run
#   bench/queue/NN-name.failed  it exited non-zero; HALTS the queue until cleared
#   bench/queue/NN-name.skipped deliberately stepped over
#   bench/queue/NN-name.log     its output
#
# Re-running `queue.sh run` after a reboot picks up at the first pass with
# neither marker. Passes should also pass `--resume` to bench/run.py so a pass
# that died halfway does not redo the cells it already recorded.
#
#   bench/queue.sh status bench/queue/2026-08-competitor-campaign
#   bench/queue.sh run    bench/queue/2026-08-competitor-campaign
#   bench/queue.sh install bench/queue/...   LaunchAgent: work it at every login
#   bench/queue.sh uninstall
#   bench/queue.sh retry 05-pass2-competitors   clear a .failed marker
set -uo pipefail

REPO="${0:A:h:h}"
# A campaign is a directory of passes: bench/queue/2026-08-competitor-campaign.
# Pass it as the last argument, or set MLXCAT_QUEUE_DIR.
QUEUE="${MLXCAT_QUEUE_DIR:-$REPO/bench/queue}"
LOCK="$QUEUE/.lock"
LABEL="com.local-ai-cat.mlxcat-bench-queue"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

log() { print -r -- "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

cmd_run() {
  mkdir -p "$QUEUE"
  # One worker at a time. A stale lock from a panic is reclaimed: the reboot
  # cleared the pid, and refusing to run because a dead run held the lock is
  # exactly the failure this script exists to remove.
  if [[ -d "$LOCK" ]]; then
    local holder; holder=$(cat "$LOCK/pid" 2>/dev/null || echo "")
    if [[ -n "$holder" ]] && kill -0 "$holder" 2>/dev/null; then
      log "queue already being worked by pid $holder — nothing to do"; return 0
    fi
    log "reclaiming stale lock from pid ${holder:-unknown} (not running)"
    rm -rf "$LOCK"
  fi
  mkdir "$LOCK" || { log "could not take the lock"; return 1; }
  print -r -- "$$" > "$LOCK/pid"
  trap 'rm -rf "$LOCK"' EXIT INT TERM

  local pass base ran=0
  for pass in "$QUEUE"/[0-9]*.sh(N); do
    base="${pass:r}"
    [[ -f "$base.done" ]] && { log "skip ${pass:t} (done)"; continue; }
    [[ -f "$base.skipped" ]] && { log "skip ${pass:t} (marked skipped)"; continue; }
    # A failed pass HALTS the queue rather than being stepped over. Passes are
    # ordered because later ones build on earlier ones — pass 3 rebuilds the
    # binaries pass 4 measures — so quietly continuing past a failure produces
    # rows that look fine and were measured against the wrong thing.
    if [[ -f "$base.failed" ]]; then
      log "HALT at ${pass:t}: $(cat "$base.failed"). Fix it and 'queue.sh retry ${pass:t:r}', or"
      log "     'queue.sh skip ${pass:t:r}' if later passes genuinely do not need it."
      return 1
    fi
    log "START ${pass:t}"
    ran=1
    # Logs append so a campaign keeps its history, but a re-run's output butting
    # straight onto the previous attempt's is how a stale "Ran 19 tests / OK"
    # got read as this run's verdict on 2026-08-22. Mark the boundary.
    print -r -- "" >>"$base.log"
    print -r -- "===== run $(date -u +%Y-%m-%dT%H:%M:%SZ) @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) =====" >>"$base.log"
    if zsh "$pass" >>"$base.log" 2>&1; then
      date -u +%Y-%m-%dT%H:%M:%SZ > "$base.done"
      log "DONE ${pass:t}"
    else
      local rc=$?
      print -r -- "rc=$rc at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$base.failed"
      log "FAILED ${pass:t} rc=$rc — stopping, see ${base:t}.log"
      # Stop rather than march on: passes are ordered because later ones depend
      # on earlier builds, and a failure that took the host down should not be
      # followed straight into the next one.
      return "$rc"
    fi
  done
  (( ran )) || log "queue empty — nothing pending"
}

cmd_status() {
  [[ -d "$QUEUE" ]] || { print "no queue at $QUEUE"; return 0; }
  local pass base state
  for pass in "$QUEUE"/[0-9]*.sh(N); do
    base="${pass:r}"
    if [[ -f "$base.done" ]]; then state="done    $(cat "$base.done")"
    elif [[ -f "$base.failed" ]]; then state="FAILED  $(cat "$base.failed")"
    elif [[ -f "$base.skipped" ]]; then state="skipped $(cat "$base.skipped")"
    else state="pending"; fi
    printf '%-40s %s\n' "${pass:t}" "$state"
  done
  if [[ -d "$LOCK" ]]; then
    local holder; holder=$(cat "$LOCK/pid" 2>/dev/null)
    kill -0 "$holder" 2>/dev/null && print "\nworking now: pid $holder" || print "\nstale lock (pid ${holder:-?} gone) — next run reclaims it"
  fi
}

cmd_install() {
  mkdir -p "${PLIST:h}"
  cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/zsh</string><string>-lc</string><string>$REPO/bench/queue.sh run $QUEUE</string></array>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$QUEUE/.launchd.log</string>
  <key>StandardErrorPath</key><string>$QUEUE/.launchd.log</string>
</dict>
</plist>
PLIST_EOF
  launchctl bootout "gui/$UID/$LABEL" 2>/dev/null
  launchctl bootstrap "gui/$UID" "$PLIST" || { print "bootstrap failed"; return 1; }
  print "installed $LABEL — the queue is worked at every login, so a panic costs one pass, not the queue"
}

cmd_uninstall() {
  launchctl bootout "gui/$UID/$LABEL" 2>/dev/null
  rm -f "$PLIST"
  print "removed $LABEL"
}

cmd_retry() {
  local name="${1:?usage: queue.sh retry NN-name}"
  local marker="$QUEUE/${name:r}.failed"
  [[ -f "$marker" ]] || { print "no failure recorded for ${name:r} in $QUEUE"; return 1; }
  rm -f "$marker"; print "cleared ${name:r}.failed — 'queue.sh run $QUEUE' resumes there"
}

cmd_skip() {
  local name="${1:?usage: queue.sh skip NN-name}"
  local base="$QUEUE/${name:r}"
  [[ -f "$base.sh" ]] || { print "no such pass: ${name:r} in $QUEUE"; return 1; }
  rm -f "$base.failed"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$base.skipped"
  print "skipped ${name:r} — the queue will step over it"
}

action="${1:-run}"
[[ $# -gt 0 ]] && shift
case "$action" in
  run|status|install|retry|skip)
    if [[ $# -gt 0 && -d "$1" ]]; then QUEUE="${1:A}"; shift; fi
    LOCK="$QUEUE/.lock"
    ;;
esac

case "$action" in
  run) cmd_run ;;
  status) cmd_status ;;
  install) cmd_install ;;
  uninstall) cmd_uninstall ;;
  retry) cmd_retry "$@" ;;
  skip) cmd_skip "$@" ;;
  *) print "usage: queue.sh {run|status|install|retry|skip} [campaign-dir] [NN-name] | uninstall"; exit 64 ;;
esac
