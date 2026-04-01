# ts-benchmark-official-adapters

Official multivariate model adapters shipped alongside the `ts-benchmark`
repo as a separate Python project.

This subproject exposes three discoverable plugins:

- `gluonts_deepvar`
- `gluonts_gpvar`
- `pytorchts_timegrad`

The core benchmark remains the `ts-benchmark` library. This package is a
separate installable plugin project in the same repo so heavyweight backend
dependencies can evolve independently from the benchmark core.

## Install

Install both projects into the same environment:

```bash
python -m pip install -e .
python -m pip install -e official_adapters
```

Install with all optional backends:

```bash
python -m pip install -e official_adapters[all]
```

Or install only the backend group you need:

```bash
python -m pip install -e official_adapters[gluonts-mx]
python -m pip install -e official_adapters[timegrad]
```

For the current `pytorchts` release line, use `setuptools<81` in the
environment. The adapters include compatibility patches for published
`pytorchts 0.6.x`, GluonTS, and MXNet APIs needed on Python 3.12.

## Recommended Env Layout

The benchmark supports per-model external execution. A practical split is:

- main benchmark env: built-ins, notebook work, active development models
- dedicated `timegrad` env: `pytorchts_timegrad`
- dedicated `eqbench-mxnet` env: `gluonts_deepvar`, `gluonts_gpvar`

That keeps heavyweight adapter stacks isolated while still exposing those
models as normal benchmark plugins. The notebook API can provision the
TimeGrad env for you with `ts_benchmark.notebook.provision_adapter_venv(...)`.

Example TimeGrad model block:

```json
{
  "name": "timegrad_external",
  "reference": {
    "kind": "plugin",
    "value": "pytorchts_timegrad"
  },
  "execution": {
    "mode": "subprocess",
    "venv": "timegrad"
  },
  "params": {
    "epochs": 1,
    "batch_size": 8
  },
  "pipeline": "raw"
}
```

Example model block:

```json
{
  "name": "deepvar_external",
  "reference": {
    "kind": "plugin",
    "value": "gluonts_deepvar"
  },
  "execution": {
    "mode": "subprocess",
    "venv": "eqbench-mxnet"
  },
  "params": {
    "epochs": 1,
    "batch_size": 8
  },
  "pipeline": "raw"
}
```

## Plugin Names

Use these names in benchmark configs:

- `gluonts_deepvar`
- `gluonts_gpvar`
- `pytorchts_timegrad`

## Add A New Adapter

1. Add the adapter under `official_adapters/src/ts_benchmark_official_adapters/`.
2. Add a factory function in `plugin.py`.
3. Add the adapter metadata to `official_adapters/src/ts_benchmark_official_adapters/ts_benchmark_plugin.toml`.
4. Register the builder entry point in `official_adapters/pyproject.toml` and include the TOML file as package data.
5. Reinstall the subproject in editable mode and verify discovery.

Example entry points:

```toml
[project.entry-points."ts_benchmark.models"]
my_model = "ts_benchmark_official_adapters.plugin:build_my_model"

[tool.setuptools.package-data]
ts_benchmark_official_adapters = ["ts_benchmark_plugin.toml"]
```

Verification:

```bash
python -m pip install -e official_adapters
ts-benchmark plugins --json
ts-benchmark validate /path/to/config.json
```
