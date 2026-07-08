# Best Practices for Autonomous Claude Code Agents — Reference

Research reference for building **PanoPin**, a project developed primarily by an
autonomous Claude Code agent across many separate sessions. All practices below are
drawn from primary Anthropic sources (engineering blog + official Claude Code docs),
which I fetched and read directly; see **Sources**. Where a phrasing is Anthropic's
own it is quoted. The design bias for PanoPin: **sub-agents, deterministic/rule-based
scaffolding, and tight verification loops.**

The single organizing constraint behind almost every practice: *"Claude's context
window fills up fast, and performance degrades as it fills."* Models also suffer
**"context rot"** — as tokens accumulate, recall degrades — because a transformer has a
finite **"attention budget."** Manage context as the scarcest resource.

---

## 1. CLAUDE.md — the living instruction file

- **Only include what Claude can't infer and would otherwise re-explain.** Bash commands
  it can't guess, non-default code-style rules, test runners, repo etiquette, architectural
  decisions, environment quirks, and non-obvious gotchas. Exclude anything Claude can read
  from the code, standard conventions, changing details, and self-evident advice like "write
  clean code."
- **Keep it short and structured.** Target **under 200 lines**; longer files "consume more
  context and reduce adherence." Use markdown headers + bullets. Conciseness test for every
  line: *"Would removing this cause Claude to make mistakes?"* If not, cut it. *"Bloated
  CLAUDE.md files cause Claude to ignore your actual instructions!"*
- **Be specific and verifiable.** "Use 2-space indentation" beats "format code properly";
  "Run `npm test` before committing" beats "test your changes." You can raise adherence with
  emphasis ("IMPORTANT", "YOU MUST").
- **Know the hierarchy (loaded broad → specific each session):** managed policy →
  user (`~/.claude/CLAUDE.md`) → project (`./CLAUDE.md` or `./.claude/CLAUDE.md`) →
  local (`./CLAUDE.local.md`, gitignored). Parent-dir files load in full at launch; child-dir
  files load on demand when Claude reads files there.
- **Use `@path` imports** to pull in additional files (README, package.json, topic guides;
  recursive up to four hops) — but imports still load into context at launch, so they aid
  organization, not context savings. For instructions that matter only for some files, use
  **path-scoped `.claude/rules/*.md`** (YAML `paths:` frontmatter) so they load only when
  matching files are touched.
- **Treat it like code and iterate.** Run `/init` to bootstrap, then refine. Add an entry
  when Claude repeats a mistake, when review catches something it should have known, or when
  you retype the same correction. Prune when Claude ignores a rule (file too long) or asks
  something already answered (phrasing ambiguous). *"The file compounds in value over time."*

---

## 2. Cross-session memory / avoiding context loss

- **Persist state to durable artifacts, because each session starts with a fresh context
  window and "compaction isn't sufficient" for long-horizon work.** The long-running-agent
  harness relies on: **git history** with descriptive commits (enables reverting bad changes),
  a **progress file** (`claude-progress.txt`) documenting what was done, and a **structured
  task list** tracking status.
- **Adopt a fixed session-startup ritual:** read the progress file → check git log → run the
  basic end-to-end test → pick the next unit of work. This lets a fresh context "quickly
  understand the state of work."
- **Structured note-taking:** *"The agent regularly writes notes persisted to memory outside
  of the context window,"* pulled back in later (e.g. `NOTES.md`, to-do lists, maps). This
  gives multi-hour/multi-session coherence across context resets.
- **Prefer JSON for machine-tracked state** (e.g. a `features.json` with `passes: false`
  flags): *"the model is less likely to inappropriately change JSON files"* than markdown.
- **Compaction as a lever, not a crutch:** when nearing the limit, summarize and reinitialize,
  **preserving "architectural decisions, unresolved bugs, and implementation details"** and
  discarding redundant tool output. Between unrelated tasks, `/clear` outright.
- **Auto-memory (`~/.claude/projects/<project>/memory/MEMORY.md`)** is available: a concise
  index (first 200 lines / 25KB) loads every session, with topic files read on demand. Keep
  the index lean and push detail into topic files.

---

## 3. Sub-agent orchestration

- **Fan out when a side task would flood the main context** with search results, logs, or file
  contents you won't reuse. The sub-agent works in an isolated window and returns only a
  **"condensed, distilled summary of its work (often 1,000–2,000 tokens)"** — detailed
  exploration stays out of the lead agent's context.
- **Each sub-agent starts fresh and self-contained.** It does *not* see conversation history,
  prior file reads, or already-invoked skills — only its system prompt + the delegation message
  (+ CLAUDE.md/memory, except the built-in Explore/Plan agents). So **brief it fully**: name the
  files/interfaces, restate any rule it must honor, and state exactly what to return.
- **Design focused, single-purpose agents with detailed descriptions and minimal tools.**
  Anthropic's own best-practice list: *"each subagent should excel at one specific task,"*
  *"write detailed descriptions,"* *"limit tool access"* (e.g. deny Write/Edit for a reviewer),
  and *"check into version control."*
- **Parallelize only independent investigations** ("research the auth, DB, and API modules in
  parallel"); **chain** them for dependent multi-step flows (reviewer → optimizer). Beware that
  many sub-agents each returning detailed results can itself fill the main context.
- **Use a sub-agent to verify** — see §5. A reviewer in a fresh context "sees only the diff and
  the criteria you give it," so the agent doing the work isn't the one grading it.
- **Give recurring agents persistent memory** (`memory: project`) so a reviewer/researcher
  accumulates codebase patterns across sessions; `project` scope is the recommended default
  (shareable via version control).

---

## 4. Deterministic / simplicity-first design

- **Find the simplest thing that works; add complexity only when it demonstrably helps.**
  Agentic systems "trade latency and cost for better task performance" — for many tasks a single
  well-prompted LLM call with good context is enough.
- **Prefer workflows (predefined code paths) over open-ended agentic loops when the task is
  well-defined.** *"Workflows offer predictability and consistency for well-defined tasks,
  whereas agents are the better option when flexibility and model-driven decision-making are
  needed at scale."* Reserve orchestrator-workers/agent loops for tasks whose sub-steps can't be
  predicted up front.
- **Workflow patterns to reach for (in rough order of complexity):** *prompt chaining*
  (sequential steps with programmatic gates), *routing* (classify then dispatch), *parallelization*
  (sectioning or voting), *orchestrator-workers* (a lead dynamically decomposes and delegates),
  *evaluator-optimizer* (generate → critique → refine against clear criteria).
- **Use deterministic scaffolding instead of trusting the loop:** an `init.sh` for reproducible
  setup, a JSON task list, hooks for actions that must happen every time (hooks are enforced;
  CLAUDE.md is only advisory). This prevents "one-shotting" and premature victory declarations.
- **Three core implementation principles:** maintain **simplicity**, prioritize **transparency**
  (show planning steps), and invest in the **agent-computer interface (ACI)** — treat tool
  descriptions like docstrings, give the model room to think, and Poka-yoke away misuse (e.g.
  *require absolute filepaths* so relative-path mistakes are impossible).

---

## 5. Verification loops / evals

- **Give the agent a check it can run itself.** *"Give Claude a check it can run: tests, a build,
  a screenshot to compare. It's the difference between a session you watch and one you walk away
  from."* Without a runnable check, *"looks done"* is the only signal and the human becomes the
  verification loop.
- **Provide a fixed target with concrete acceptance criteria** — example test cases, expected
  outputs, a diff-against-fixture, or a design screenshot to match. The long-running harness
  encodes this as a task list where each item has explicit end-to-end steps and a `passes: false`
  flag until proven.
- **Test-first / reproduce-then-fix:** for bugs, "write a failing test that reproduces the issue,
  then fix it." Address root causes, not symptoms; never suppress errors.
- **Protect the target: tests are ground truth, not something to edit away.** *"It is unacceptable
  to remove or edit tests."* Real verification needs real execution — Claude will "mark features as
  complete without proper testing" unless forced to actually run them (e.g. browser automation for
  UI).
- **Have the agent show evidence, not assertions** — the command it ran and its output, or a
  screenshot. Reviewing evidence is faster than re-running the check yourself and works for
  unattended sessions.
- **Add an independent, adversarial review before "done."** A fresh sub-agent reviews the diff
  against the plan so the author isn't the grader. But: *"Tell the reviewer to flag only gaps that
  affect correctness or the stated requirements, and treat the rest as optional"* — a gap-hunting
  reviewer always finds some, and chasing all of them causes over-engineering. Gate hard with a
  Stop hook or a `/goal` condition when running unattended.

---

## 6. Skills & tools usage

- **Package repeatable procedures as Skills, not CLAUDE.md bloat.** *"Create a skill when you keep
  pasting the same instructions, checklist, or multi-step procedure … or when a section of
  CLAUDE.md has grown into a procedure rather than a fact."* A skill's body loads **only when
  used**, so "long reference material costs almost nothing until you need it."
- **Progressive disclosure is the whole point:** a skill's `description` is always in context so
  Claude knows it exists; the full `SKILL.md` body loads on invocation; bulky reference files/scripts
  in the skill directory load only when referenced. Keep `SKILL.md` **under 500 lines** and push
  detail into supporting files.
- **Skill vs sub-agent:** a skill runs **inline in the main conversation** (reusable prompt/knowledge);
  a sub-agent runs in an **isolated context** and returns a summary. Use `disable-model-invocation:
  true` for side-effecting workflows (`/deploy`, `/commit`) so Claude can't trigger them on its own.
- **Prefer CLI tools and give the agent thinking space.** CLI tools (`gh`, `aws`, `gcloud`) are the
  most context-efficient way to hit external services; install `gh` so Claude isn't rate-limited on
  unauthenticated calls. Follow ACI hygiene: clear tool docs with examples/edge cases, formats close
  to what the model has seen naturally, and error-proofed inputs.
- **Match the mechanism to the need:** facts every session → CLAUDE.md; procedures on demand → skill;
  isolation/verbose output → sub-agent; must-happen-every-time enforcement → hook.
- **Evaluate skills with a baseline comparison:** run realistic prompts with the skill available vs.
  disabled, in fresh sessions, and compare — measure both *does it trigger* and *is the output right*.

---

## Sources

All URLs below were fetched and read for this document (primary Anthropic sources only).

1. **Best practices for Claude Code** — https://code.claude.com/docs/en/best-practices
   *(the Anthropic engineering URL `anthropic.com/engineering/claude-code-best-practices` 308-redirects here)*
2. **Building Effective AI Agents** — https://www.anthropic.com/engineering/building-effective-agents
3. **Effective context engineering for AI agents** (Anthropic Engineering, Sep 29 2025) —
   https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
4. **Effective harnesses for long-running agents** (Anthropic Engineering, Nov 26 2025) —
   https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
5. **Create custom subagents** (Claude Code docs) — https://code.claude.com/docs/en/sub-agents
6. **How Claude remembers your project** (Claude Code memory docs) — https://code.claude.com/docs/en/memory
7. **Extend Claude with skills** (Claude Code docs) — https://code.claude.com/docs/en/skills

*Note:* the task allowed 1–2 secondary sources, but the seven primary Anthropic sources above
fully cover all six areas, so no secondary sources were used (per the "prioritize primary" and
"cite only what you retrieved" constraints).

---

## Applied to PanoPin — the ~10 rules an autonomous agent should follow

Mapping the highest-leverage practices to concrete project rules for a multi-session, autonomous build:

1. **Keep `PanoPin/CLAUDE.md` under ~200 lines**, facts-only (build/test commands, layout, conventions,
   gotchas). Run the conciseness test on every line; prune whenever a rule stops being obeyed. *(§1)*
2. **Maintain a `docs/progress.md` (or `claude-progress.txt`) and a `docs/decisions.md`.** Update both at
   the end of every session; they are the memory that survives a fresh context window. *(§2)*
3. **Start every session with the same ritual:** read progress + decision logs → `git log` → run the
   smoke/e2e test → pick the next open task. Only then write code. *(§2, §5)*
4. **Track work in a machine-readable `tasks.json`** where each item has explicit verification steps and a
   `passes: false` flag flipped only when the check actually runs green. *(§2, §5)*
5. **Commit frequently with descriptive messages on a branch**, so any bad change is one `git revert` away;
   never work directly on the default branch unattended. *(§2)*
6. **Every task must ship with a runnable check** (unit test, build, script that diffs against a fixture).
   No check ⇒ not done. Show the command output as evidence, don't assert success. *(§5)*
7. **Tests are ground truth — never edit or delete a test to make a task pass.** Fix root causes; reproduce
   bugs with a failing test first. *(§5)*
8. **Fan out research/verbose work to sub-agents** (isolated context, minimal tools, detailed brief that
   names files and restates constraints); pull back only a short summary. Use the built-in Explore agent
   for read-only codebase search. *(§3)*
9. **Before marking any non-trivial task done, run an adversarial review in a fresh sub-agent** against the
   task's stated requirements — flag only correctness/requirement gaps, not style. *(§5, §3)*
10. **Prefer the simplest mechanism: deterministic workflow over open-ended loop; hook for must-happen
    steps; skill for repeatable procedures (loads on demand); sub-agent for isolation.** Add agentic
    complexity only when it demonstrably beats a simpler path. *(§4, §6)*
11. **Package recurring multi-step procedures as `.claude/skills/*/SKILL.md`** (≤500 lines, progressive
    disclosure) instead of growing CLAUDE.md; guard side-effecting ones with `disable-model-invocation`. *(§6)*
12. **Between unrelated tasks, `/clear`; when compacting, preserve architectural decisions, open bugs, the
    modified-file list, and test commands.** Treat context as the scarcest resource. *(§2)*
