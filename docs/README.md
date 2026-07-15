# HandUMI documentation

This directory contains the source for the HandUMI documentation website.
The site is built with Sphinx, MyST Markdown, and the Sphinx Book Theme.

## Build locally

Create an isolated documentation environment from the repository root:

```bash
uv venv .venv-docs
uv pip install --python .venv-docs/bin/python -r docs/requirements.txt
```

Build the HTML site:

```bash
PATH="$PWD/.venv-docs/bin:$PATH" make -C docs html
```

The generated site is written to `docs/build/html/`. Preview it with:

```bash
python -m http.server --directory docs/build/html 8000
```

Then open <http://localhost:8000>.

## Validation

The default build treats Sphinx warnings as errors. To check external and
internal links as a separate step, run:

```bash
PATH="$PWD/.venv-docs/bin:$PATH" make -C docs linkcheck
```

Generated files under `docs/build/` are intentionally not committed.

## GitHub Pages

Pushing documentation changes to `main` triggers
`.github/workflows/docs.yml`. It builds the Sphinx site and deploys it to:

<https://robonet-ai.github.io/handumi-sw/>

In the repository settings, set **Pages → Build and deployment → Source** to
**GitHub Actions**.
