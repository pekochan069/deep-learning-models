# Deep Learning Models

## Requirements

[uv](https://docs.astral.sh/uv/) is required to run the models. Install it using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # Linux
winget install astral-sh.uv # Windows
```

Change name of `pyproject-{platform}.toml` to `pyproject.toml` and `uv-{platform}.lock` to `uv.lock` (`pyproject-cuda.toml` to `pyproject.toml` and `uv-cuda.lock` to `uv.lock`)

```bash
uv sync
```

## Run

```bash
uv run poe main
```

## Scripts

| Script | Description |
| ------ | ----------- |
| `uv run poe main` | Runs the main script using the Poe model. |
| `uv run poe format` | Formats the source code using ruff. |
| `uv run poe lint` | Lints the source code using ruff. |
