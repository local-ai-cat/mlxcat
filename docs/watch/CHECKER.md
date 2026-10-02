# Watch checker: one weekly pass

You are keeping [`docs/ENGINES.md`](../ENGINES.md) and
[`LEARNINGS.md`](LEARNINGS.md) current. Do one pass, then stop. You need `gh`
(authenticated, read access is enough) and `python3`. Expect about an hour.

## Rules

- **Read only, outside this repo.** Never open, comment on, react to, star,
  fork or watch anything on another repository. `gh api` GET calls and shallow
  clones into `guest/` (gitignored) only.
- **Learn mechanisms, don't copy code.** Copy only from Apache-2.0-compatible
  licences, and record provenance in `NOTICE` and the commit message if you do.
- **No runtime changes.** This pass writes docs and the cursor file. It does not
  build, benchmark, bump a pin or touch `Sources/`.
- **One pass.** If `gh` fails, retry once. If it fails again, stop with a
  `blocked` verdict.
- **Public voice.** This repo is public. Describe other projects fairly and
  factually. No names of people, no local paths, no hostnames.

## Steps

1. **Drift.** Run `scripts/donor-drift.sh > /tmp/drift.md`. Each watched repo is
   reported from its cursor in [`cursors.json`](cursors.json): commits newer than
   the last one a pass read, plus `(new)` next to a release that appeared since.
2. **Pick what to read.** Every repo with commits or a new release, in this order:
   pinned dependencies (the top table), then repos whose "why" says
   *competitor*, then the rest. Read at most **12 repos**. Carry the remainder
   to next week by leaving their cursors where they are.
3. **Read each one** from its cursor date: the release notes, then merged PRs
   (`gh api 'search/issues?q=repo:OWNER/REPO+is:pr+is:merged+merged:>=DATE&per_page=100'`),
   then the code, if the PR body does not explain the mechanism. Skim titles.
   Read the body of anything about prefill, decode, batching, scheduling, KV
   cache, memory, speculative decoding, tokenizers/templates or tool calling.
4. **Check mlxcat for each candidate before writing it down.** `grep` the
   mechanism in `Sources/` and cite `file:line` for "has it", "partial" or "no".
   An entry that guesses about mlxcat is worse than no entry.
5. **Append to `LEARNINGS.md`** under a new `## Pass YYYY-MM-DD` heading, using
   the entry format at the top of that file. Every entry needs a link (PR,
   commit or release). A repo you read with nothing relevant gets one line:
   `Nothing relevant since <cursor date> (read: …).`
6. **Advance the cursor** for each repo you actually read. Set `commit` to the
   HEAD you read up to, plus `commit_date`, `release`, `read_on` (today) and
   `depth` (`read` or `skimmed`). Leave unread repos untouched.
7. **Propose at most two ports.** Under the pass heading, add `### Proposed ports`
   with at most two learnings marked **port now**. Give each the lever it would
   become under [`docs/LEVERS.md`](../LEVERS.md) (off by default, its exit, and
   how to price it). Do not implement them.
8. **Bump `last_verified`** in the `living:` block of `docs/ENGINES.md` to today.
   If a watched repo was archived, renamed or went quiet for 180 days, note it
   under the pass heading. Do not edit the watchlist table yourself; discovery
   and removals are a separate, monthly pass (see `DISCOVERY.md`).
9. **Commit once** on a branch `watch/YYYY-MM-DD`:
   `docs(watch): weekly pass YYYY-MM-DD — N learnings, M repos read`, with the
   trailer `Agent: <agent name and model>`. Push the branch only if you were told
   to. Never merge it yourself.
10. **Print the verdict** as your last line and stop:
    `VERDICT watch: current` (nothing relevant) ·
    `VERDICT watch: N learnings, M ports proposed` ·
    `VERDICT watch: blocked <why>`.

## Monthly add-on: discovery

Once a month, after the weekly steps, run the methods in
[`DISCOVERY.md`](DISCOVERY.md) that are marked worth repeating. Apply its entry
test and add at most five repos. A new repo enters the drift report by getting a
`baseline` entry in `cursors.json` (HEAD today, `depth: baseline`) and a row in
the `docs/ENGINES.md` watchlist.
