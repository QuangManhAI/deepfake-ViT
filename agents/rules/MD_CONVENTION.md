# MD_CONVENTION.md

Rules for any .md file the agent creates in this project. Condensed from
MD_creation_guide.md — see that file for full rationale and examples.

## Required header (5 fields, in order)

```
# <Title>

- **Motivation/Background**: 1–3 sentences — why this doc exists.
- **Purpose**: one sentence — what this doc achieves.
- **Overview Pipeline**: 1–2 sentences — the process that produced this content.
- **Detailed Plan**: compact list of sections/subsections and what each covers.
- **References**: comma-separated libraries/tools/frameworks used.

---
```

## Table of Contents

- Place immediately after the header `---`, before first content section
- Include all `##` and `###` headings; omit `####`+ unless critical
- Anchor rule: lowercase, strip punctuation except hyphens, spaces→hyphens
- Regenerate TOC whenever a section is added/removed

## Mandatory cross-reference links

- **Whenever a file, module, notebook, or artifact path is mentioned in
  AI-generated Markdown, it MUST be a working cross-reference link** — never
  bare text. This keeps documentation verifiable and drift-free.
- Link form:
  - Same tree: relative link from the document's location
    (`src/training/train_model.py`, `notebooks/<analysis>.ipynb`).
  - Jump to a heading: cross-file anchor link
    (`agents/rules/LOGGING_CHECKPOINT_RULES.md#5-resume-procedure`).
  - External resources: absolute URL.
- Rules and phase/progress docs must additionally cross-link the rule they
  enforce (e.g. any logging-related doc links
  [LOGGING_CHECKPOINT_RULES.md](LOGGING_CHECKPOINT_RULES.md)).
- Notebook headers must include a `## References` block linking the rules,
  scripts, and artifact locations they consume — see
  [NOTEBOOK_HEADER_CONVENTION.md](NOTEBOOK_HEADER_CONVENTION.md).

## Body formatting

- `##` top-level sections, `###` subsections, `####` only if necessary
- No vague headings ("Details", "Info")
- Tables for structured comparisons; bullets for unordered items;
  numbered lists for sequential steps; blockquotes for experiment-design
  notes; bold for key metrics; code blocks for paths/commands/config
- Each experiment section needs a `### Key findings` subsection:
  plain-language summary, connects to prior experiments, flags regressions
- Multi-experiment docs end with `## Cross-Experiment Summary`
  (comparison table + narrative + limitations)

## Conventions

| Rule                      | Requirement                                                                                                                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Single-variable principle | One changed factor per experiment; state what's held constant                                                                                                                                       |
| Cross-reference links     | **Mandatory** — any file/notebook/path mention must be a working relative link (see [Mandatory cross-reference links](#mandatory-cross-reference-links))                                            |
| File paths                | Relative link (`agents/OVERVIEW.md`: internal), Cross-file anchor link (`agents/OVERVIEW.md#installation`: jump to a heading), Absolute URL (external) — never bare text                             |
| Notebook refs             | `notebooks/<category>/<experiment>/<filename>.ipynb`; link the notebook AND its consuming scripts/artifacts                                                                                         |
| Output dirs               | `experiments/runs/<ts>_<run>/` (run state), `experiments/results/<experiment>/` (consolidated outputs) — see [LOGGING_CHECKPOINT_RULES.md](LOGGING_CHECKPOINT_RULES.md)                              |
| Notebook headers          | First cell per [NOTEBOOK_HEADER_CONVENTION.md](NOTEBOOK_HEADER_CONVENTION.md): title, subtitle, roadmap table, **and `## References`** with links to rules/scripts/artifacts                          |
| Results (5W1H)            | Every reported metric carries full 5W1H context — see [RESULTS_REPORTING.md](RESULTS_REPORTING.md)                                                                                                  |
| Dates                     | `YYYY-MM-DD`                                                                                                                                                                                      |
| Separators                | `---` after header and between major sections                                                                                                                                                     |
| Variable docs             | `> **Variable changed**: ...` / `> **Held constant**: ...` pair above each experiment                                                                                                           |
| Metric deltas             | Show Δ columns in comparison tables; bold positive Δ                                                                                                                                              |

## Self-review before finalizing

- [ ] All 5 header fields present
- [ ] TOC anchors resolve
- [ ] Every file/notebook/path mention is a working cross-reference link
- [ ] Notebook headers contain `## References` links (if the doc discusses notebooks)
- [ ] Metrics match source
- [ ] File/notebook paths correct and current (no stale paths copied from another project)
