# Changelog

All notable changes to this repository are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Governance baseline: LICENSE, security policy, code of conduct, contributor and
  support guides, issue and pull-request templates, CODEOWNERS, dependabot, and
  `.editorconfig` added. No research content changed.
- Collection evidence is `make check`: offline pytest for every active project,
  committed to `results/status-matrix.json`, with the README tier table held
  equal by `scripts/drift_check.py`. Projects `12` and `16` are rows in that
  matrix. Tiers name the offline run. They do not name a production deploy.
- `12-prompt-eval-framework` dataset tests write under `tempfile` instead of
  a hardcoded `/data/user/work` path, so a fresh clone can collect them.
