# Contributing

Thanks for your interest in improving this repository.

## Getting set up

```bash
git clone https://github.com/aidless/llm-agent-projects.git
cd llm-agent-projects
python3 -m venv .venv
. .venv/bin/activate
make check-install
```

`make check-install` is the networked step (`requirements-check.txt` plus NLTK tokenizers). There is no single root `requirements.txt`; each numbered directory has its own, and the collection check uses the shared file above.

## Verifying your change

```bash
make check
```

That is the command CI runs (`.github/workflows/ci.yml`, job `offline-check`). It re-runs every active project's offline tests and fails if `results/status-matrix.json` or the README tier table would change. After a real change to those counts, run `make check-refresh` and commit the matrix.

## Pull requests

- One logical change per pull request.
- Explain what breaks or is missing without the change.
- State how you verified it, and paste the command output.
- Keep unrelated reformatting out of the diff.

## Security

Do not open a public issue for a security problem. Use the private channel described
in [SECURITY.md](SECURITY.md).

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
