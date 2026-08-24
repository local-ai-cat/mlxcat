#!/bin/zsh
# donor-drift.sh — how far have the repos we pin / port from / compete with moved?
#
# Reads the pins from Package.swift (revision: / tag), asks the GitHub API for
# each repo's default-branch HEAD, newest release, and the commits since our pin
# (or since a window for watchlist repos), and prints a Markdown report. Keyword
# hits in commit subjects are flagged so a reader can triage in one pass.
#
#   scripts/donor-drift.sh                 # markdown to stdout
#   scripts/donor-drift.sh --json          # machine-readable
#   DRIFT_SINCE_DAYS=14 scripts/donor-drift.sh
#
# Needs: gh (authenticated) or GITHUB_TOKEN, python3. No builds, no clones.
# Footprint rule: this script only READS public repos; it never opens issues or
# PRs anywhere. The weekly workflow upserts an issue in THIS repo only.

set -euo pipefail
SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"
FORMAT="md"
[[ "${1:-}" == "--json" ]] && FORMAT="json"

export DRIFT_FORMAT="$FORMAT"
export DRIFT_REPO_ROOT="$REPO_ROOT"
export DRIFT_SINCE_DAYS="${DRIFT_SINCE_DAYS:-30}"

exec python3 - <<'PY'
import json, os, re, subprocess, sys, datetime as dt
from pathlib import Path

ROOT = Path(os.environ["DRIFT_REPO_ROOT"])
FMT = os.environ["DRIFT_FORMAT"]
SINCE_DAYS = int(os.environ["DRIFT_SINCE_DAYS"])
KEYWORDS = ["cache", "prefill", "memory", "batch", "rope", "gemma", "qwen", "kv", "stream", "metal", "quant", "mlx", "speculative", "tokenizer", "whisper"]

# ---- pins from Package.swift --------------------------------------------- #
pkg = (ROOT / "Package.swift").read_text()
pins = []
for m in re.finditer(r'url:\s*"https://github.com/([^/"]+/[^/"]+?)(?:\.git)?"\s*,\s*(?:revision:\s*"([0-9a-f]{7,40})"|\.upToNextMinor\(from:\s*"([^"]+)"\)|from:\s*"([^"]+)"|exact:\s*"([^"]+)")', pkg, re.S):
    repo, rev, minor, frm, exact = m.groups()
    pins.append({"repo": repo, "kind": "revision" if rev else "tag", "pin": rev or minor or frm or exact})

# Resolved pins (more precise when present)
resolved = ROOT / "Package.resolved"
if resolved.exists():
    try:
        data = json.loads(resolved.read_text())
        for p in data.get("pins", []):
            loc = p.get("location", "")
            mm = re.search(r"github.com/([^/]+/[^/.]+)", loc)
            if not mm: continue
            repo = mm.group(1)
            for pin in pins:
                if pin["repo"].lower() == repo.lower():
                    st = p.get("state", {})
                    pin["resolved_revision"] = st.get("revision")
                    pin["resolved_version"] = st.get("version")
    except Exception:
        pass

WATCH = [
    # repo, why
    ("ml-explore/mlx", "core kernels"),
    # The next two are pinned via atlas-open-sources forks, so the pins section
    # above watches OUR fork — these rows keep upstream in view. Each fork
    # carries one fix and dies the release upstream absorbs it:
    ("ml-explore/mlx-swift", "UN-FORK when a release vendors mlx >= 76a977ca (RoPE batch-grid fix, 2026-05-11)"),
    ("ml-explore/mlx-swift-lm", "UN-FORK when upstream rotates gemma with per-row ropeOffset (our 7b93094e)"),
    ("ml-explore/mlx-lm", "batched-decode mechanism reference"),
    ("jundot/omlx", "serving-architecture reference (ported from)"),
    ("ddalcu/mlx-serve", "gotchas mining"),
    ("lmstudio-ai/mlx-engine", "disk-chunked KV / LM Studio engine"),
    ("waybarrios/vllm-mlx", "competitor: batching + paged KV on MLX"),
    ("Blaizzy/mlx-vlm", "VLM checkpoint authority / spec-decode reference"),
    ("Trans-N-ai/swama", "competitor: native Swift MLX engine (mac+iOS)"),
    ("SharpAI/SwiftLM", "competitor: native Swift MLX server"),
    ("raullenchai/Rapid-MLX", "competitor: Ollama-replacement MLX engine"),
    ("ollama/ollama", "competitor: MLX backend since v0.30"),
    ("ggml-org/llama.cpp", "GGUF/Metal baseline"),
    ("google-ai-edge/LiteRT-LM", "iOS competitor for Gemma"),
    ("vllm-project/vllm", "scheduler design source"),
    ("john-rocky/apple-silicon-llm-bench", "neutral Mac+iPhone benchmark methodology"),
]

API_ERRORS = []

def gh(path):
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    if out.returncode != 0:
        API_ERRORS.append(path)
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        API_ERRORS.append(path)
        return None

def gh_paginate(path):
    out = subprocess.run(["gh", "api", "--paginate", path], capture_output=True, text=True)
    if out.returncode != 0:
        API_ERRORS.append(path)
        return None
    items = []
    # --paginate concatenates JSON arrays; split defensively
    for chunk in re.split(r"\]\s*\[", out.stdout.strip()):
        chunk = chunk if chunk.startswith("[") else "[" + chunk
        chunk = chunk if chunk.endswith("]") else chunk + "]"
        try:
            items.extend(json.loads(chunk))
        except json.JSONDecodeError:
            pass
    return items

def head(repo):
    info = gh(f"repos/{repo}")
    if not info: return None
    branch = info.get("default_branch", "main")
    commits = gh(f"repos/{repo}/commits?sha={branch}&per_page=1") or []
    c = commits[0] if commits else {}
    return {"branch": branch, "sha": (c.get("sha") or "")[:8], "date": ((c.get("commit") or {}).get("committer") or {}).get("date", "")[:10],
            "subject": ((c.get("commit") or {}).get("message") or "").split("\n")[0][:90], "stars": info.get("stargazers_count")}

def latest_release(repo):
    rel = gh(f"repos/{repo}/releases?per_page=1") or []
    if rel:
        return {"tag": rel[0].get("tag_name"), "date": (rel[0].get("published_at") or "")[:10]}
    tags = gh(f"repos/{repo}/tags?per_page=1") or []
    return {"tag": tags[0].get("name"), "date": ""} if tags else None

def commit_date(repo, sha):
    c = gh(f"repos/{repo}/commits/{sha}")
    return (((c or {}).get("commit") or {}).get("committer") or {}).get("date", "")[:10] or None

def commits_since(repo, since_iso, branch):
    items = gh_paginate(f"repos/{repo}/commits?sha={branch}&since={since_iso}T00:00:00Z&per_page=100")
    if items is None:
        return None  # API failure ≠ zero commits — the report must show "?", never a false 0
    subjects = [((i.get("commit") or {}).get("message") or "").split("\n")[0] for i in items]
    return subjects

def md_escape(text):
    return text.replace("|", "\\|").replace("`", "'").replace("@", "@\u200b")

def show_count(n):
    return "?" if n is None else str(n)

def keyword_hits(subjects):
    hits = []
    for s in subjects:
        low = s.lower()
        kws = [k for k in KEYWORDS if k in low]
        if kws:
            hits.append((md_escape(s[:100]), kws))
    return hits

report = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "since_days": SINCE_DAYS, "pins": [], "watch": []}
for pin in pins:
    repo = pin["repo"]
    h = head(repo)
    rel = latest_release(repo)
    rev = pin.get("resolved_revision") or (pin["pin"] if pin["kind"] == "revision" else None)
    pin_date = commit_date(repo, rev) if rev else None
    if not pin_date and pin["kind"] == "tag":
        # tag date via the release list
        rels = gh(f"repos/{repo}/releases?per_page=30") or []
        for r in rels:
            if r.get("tag_name", "").lstrip("v") == str(pin["pin"]).lstrip("v"):
                pin_date = (r.get("published_at") or "")[:10]
    subjects = commits_since(repo, pin_date, h["branch"]) if (h and pin_date) else []
    report["pins"].append({**pin, "pin_date": pin_date, "head": h, "latest_release": rel,
                           "commits_since_pin": len(subjects) if subjects is not None else None,
                           "keyword_hits": keyword_hits(subjects or [])})

since = (dt.date.today() - dt.timedelta(days=SINCE_DAYS)).isoformat()
for repo, why in WATCH:
    h = head(repo)
    if not h:
        report["watch"].append({"repo": repo, "why": why, "error": "unreachable"})
        continue
    subjects = commits_since(repo, since, h["branch"])
    report["watch"].append({"repo": repo, "why": why, "head": h, "latest_release": latest_release(repo),
                            "commits_in_window": len(subjects) if subjects is not None else None,
                            "keyword_hits": keyword_hits(subjects or [])})

if FMT == "json":
    print(json.dumps(report, indent=2)); sys.exit(0)

print(f"# Donor drift report — {report['generated'][:10]}\n")
print("## Pinned dependencies (commits since OUR pin)\n")
print("| repo | pinned | pin date | upstream HEAD | newest release | commits since pin | keyword hits |")
print("|---|---|---|---|---|---:|---|")
for p in report["pins"]:
    h = p.get("head") or {}
    rel = p.get("latest_release") or {}
    if p["kind"] == "revision":
        pinned = (p.get("resolved_revision") or p["pin"])[:10]
    else:
        pinned = p["pin"] + (f" → {p['resolved_version']}" if p.get("resolved_version") else "")
    print(f"| {p['repo']} | `{pinned}` | {p.get('pin_date') or '?'} | `{h.get('sha','?')}` {h.get('date','')} | {rel.get('tag') or '—'} {rel.get('date','')} | {show_count(p['commits_since_pin'])} | {len(p['keyword_hits'])} |")
print("\n<details><summary>Keyword hits in pinned deps</summary>\n")
for p in report["pins"]:
    if p["keyword_hits"]:
        print(f"**{p['repo']}**")
        for s, kws in p["keyword_hits"][:40]:
            print(f"- {s}  _({', '.join(kws)})_")
        print()
print("</details>\n")
print(f"## Watchlist (activity in the last {SINCE_DAYS} days)\n")
print("| repo | why | HEAD | newest release | commits | keyword hits |")
print("|---|---|---|---|---:|---|")
for w in report["watch"]:
    if "error" in w:
        print(f"| {w['repo']} | {w['why']} | {w['error']} | | | |"); continue
    h = w["head"]; rel = w.get("latest_release") or {}
    print(f"| {w['repo']} | {w['why']} | `{h['sha']}` {h['date']} | {rel.get('tag') or '—'} {rel.get('date','')} | {show_count(w['commits_in_window'])} | {len(w['keyword_hits'])} |")
print("\n<details><summary>Keyword hits on the watchlist</summary>\n")
for w in report["watch"]:
    if w.get("keyword_hits"):
        print(f"**{w['repo']}**")
        for s, kws in w["keyword_hits"][:25]:
            print(f"- {s}  _({', '.join(kws)})_")
        print()
print("</details>\n")
if API_ERRORS:
    print(f"\n⚠️ {len(API_ERRORS)} GitHub API call(s) failed (rate limit / network) — '?' counts are unknown, NOT zero. First: {API_ERRORS[0]}")
print("_Generated by `scripts/donor-drift.sh`. Read-only; this report opens nothing anywhere else._")
PY
