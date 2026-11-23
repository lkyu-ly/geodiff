# Issue tracker: Local Markdown

- One feature per `.scratch/<feature-slug>/`; publish its spec as `spec.md`.
- One implementation ticket per `issues/<NN>-<slug>.md`, numbered from `01`. Fetch tickets by their referenced path or number.
- Record triage in `Status:` near the top using [triage-labels.md](triage-labels.md). Use `Blocked by: NN, NN` for dependencies. A ticket is ready only after its blockers are resolved.
- Append discussion to `## Comments`. To close implementation work, set `Status: resolved` after integration verification, and record the branch and evidence under `## Comments`.

## Wayfinding

- Use `.scratch/<effort>/map.md` as the map and numbered `issues/` files as child tickets, with `Type:` and any `Blocked by:` entries.
- Claim the first open, unblocked, unclaimed child by setting `Status: claimed`. To resolve it, append `## Answer`, set `Status: resolved`, and link it from Decisions-so-far in the map.
