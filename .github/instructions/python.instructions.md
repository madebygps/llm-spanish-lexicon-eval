---
applyTo: '**.py'
---
I like to use `uv` for dependency management, virtual environments, and running Python code. Keep it minimal and always default to `uv`.

Essential rules (concise):
- Install runtime dependency: `uv add <package>`
- Install dev / test dependency: `uv add --dev <package>`
- Run a script: `uv run main.py` (or another file / module: `uv run -m package.module`)
- Create an explicit venv only if needed: `uv venv` (usually not required because `uv run` handles it)
- Remove a dependency: `uv remove <package>`

Avoid using `pip install`, `poetry add`, `conda install`, or manual venv activation unless I explicitly say uv cannot be used.

If I ask for something like "install X" or "run Y": assume the `uv` form automatically without extra explanation.
