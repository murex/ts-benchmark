# TS Benchmark

A config-driven framework for benchmarking generative models on multivariate time-series return scenarios.

The framework is built around one core rule:

**the benchmark owns the dataset, the train/test split, the protocol definition, the rolling window extraction policy, the preprocessing pipeline, and the runtime selection; models only implement the generation logic.**

That makes runs easier to reproduce and makes model comparisons much fairer.

---

## What is included

- synthetic regime-switching factor stochastic-volatility scenario generation
- external dataset support from **CSV** and **Parquet** files
- benchmark-owned branched protocol settings:
  - common: `kind`, `horizon`, `n_model_scenarios`, `n_reference_scenarios`
  - forecast: `forecast.train_size`, `forecast.test_size`, `forecast.context_length`, `forecast.eval_stride`, `forecast.train_stride`
  - unconditional windowed: `unconditional_windowed.train_size`, `unconditional_windowed.test_size`, `unconditional_windowed.eval_stride`, `unconditional_windowed.train_stride`
  - unconditional path dataset: `unconditional_path_dataset.n_train_paths`, `unconditional_path_dataset.n_realized_paths`
- explicit preprocessing pipelines per model
- built-in models:
  - historical bootstrap
  - historical bootstrap with stochastic volatility
- plugin discovery for external model packages
- plugin manifests and capability metadata for UI / CLI discovery
- CLI runner
- notebook API for browsing benchmarks, swapping datasets, injecting local models, and inspecting results
- Streamlit UI with dataset and device selection
- metrics and saved benchmark artifacts tagged with dataset metadata

---

## Why the framework is structured this way

A benchmark is most useful when the comparison contract is stable.

In this project, the stable pieces are owned by the benchmark:

- dataset loading
- split logic
- rolling evaluation origins
- context length and forecast horizon
- supervised training-window extraction stride
- preprocessing visibility
- metrics
- reporting and artifact storage

The model side is intentionally small:

- external model authors implement the structural contract in `ts_benchmark.model_contract`
- notebook users work through `ts_benchmark.notebook`
- optionally publish a plugin manifest so the UI and CLI can describe the model clearly

This lets a model under development live in its own repository and still be benchmarked against all existing models.

---

## Installation

From the project root:

```bash
python -m pip install -e .
```

Optional PyTorch-backed helpers:

```bash
python -m pip install -e .[torch]
```

Optional UI dependencies:

```bash
python -m pip install -r requirements-ui.txt
```

Optional MLflow tracking dependencies:

```bash
python -m pip install -e .[tracking]
```

Official adapter subproject:

```bash
python -m pip install -e ./official_adapters
```

TimeGrad backend dependencies for the official adapter package:

```bash
python -m pip install -e ./official_adapters[timegrad]
```

The core `ts-benchmark` package is intentionally installable without PyTorch.
That keeps benchmark browsing, config loading, dataset inspection, metrics,
results, notebook workflows, and CPU-only built-in baselines lightweight.
Install the optional `torch` extra only when you need PyTorch-backed helpers or
device-aware acceleration in the core package.
Likewise, install official adapter backend extras only when you actually want
to execute those adapter models in a given environment. In notebook workflows,
the recommended path is often to keep the main notebook env light and use
`ts_benchmark.notebook.provision_adapter_venv(...)` to create a dedicated
subprocess environment for heavyweight adapters such as TimeGrad.

The repo is structured as a small monorepo:

- root project: `ts-benchmark`
- sibling plugin project: `official_adapters/` (`ts-benchmark-official-adapters`)

---

## Quick start

Validate a config:

```bash
ts-benchmark validate smoke_test
```

Run a benchmark:

```bash
ts-benchmark run synthetic_basic_benchmark
```

List built-in and discovered model plugins:

```bash
ts-benchmark plugins
```

List them as JSON, including manifests and capability metadata:

```bash
ts-benchmark plugins --json
```

Mutable UI/notebook workspace state is stored under
`$TS_BENCHMARK_HOME` when set, otherwise under the XDG-style default
`~/.local/share/ts-benchmark/`.

Launch the UI:

```bash
streamlit run streamlit_app.py
```

---

## Public Python API

The supported public API is split by audience.

### Minimal root package

Use `import ts_benchmark` for the small high-level surface:

- `BenchmarkConfig`
- `Protocol`
- `BenchmarkResults`
- `BenchmarkDiagnostics`
- `BenchmarkRunArtifacts`
- `load_benchmark_config`
- `validate_benchmark_config`
- `dump_benchmark_config`
- `list_benchmark_summaries`
- `summarize_benchmark`
- `run_benchmark_from_config`

### Public submodules

These are also public and intended for direct use:

- `ts_benchmark.benchmark`
  Benchmark definitions, protocol, config IO, and shipped benchmark browsing.
- `ts_benchmark.run`
  Programmatic run execution and run configuration types.
- `ts_benchmark.notebook`
  Notebook-first browsing, execution, dataset injection, model injection, and result inspection.
- `ts_benchmark.results`
  Result objects and reporting helpers.
- `ts_benchmark.metrics`
  Metric config, selection, and ranking helpers.
- `ts_benchmark.model_contract`
  The public structural contract for external model authors.
- `ts_benchmark.tracking`
  Optional MLflow integration helpers.

### Internal modules

The following are importable but not part of the stable public API:

- `ts_benchmark.ui.*`
- `ts_benchmark.model.wrappers.*`
- `ts_benchmark.model.contracts`
- `ts_benchmark.run.evaluator`
- `ts_benchmark.run.storage`
- `ts_benchmark.dataset.providers.*`

### Notebook example

```python
from ts_benchmark.benchmark import list_benchmark_summaries
from ts_benchmark.notebook import (
    dataset_frame,
    entrypoint_model,
    provision_adapter_venv,
    run_benchmark,
)

benchmarks = list_benchmark_summaries()

run = run_benchmark(
    "smoke_test",
    include=["scenarios", "diagnostics"],
    with_model=entrypoint_model(
        "my_local_model",
        "/path/to/model.py:build_estimator",
        ridge=0.1,
    ),
)

metrics = run.metrics()
band = run.scenario_band("my_local_model", evaluation_window=0, asset=0)
dataset = dataset_frame("smoke_test").frame
```

For dataset-first notebook workflows that rerun heavyweight official adapters,
you can provision a dedicated subprocess env instead of installing those
dependencies into the main notebook env:

```python
timegrad_env = provision_adapter_venv(
    "outputs/venvs/timegrad",
    "pytorchts_timegrad",
)
```

---

## Benchmark-owned protocol contract

The benchmark controls:

- `protocol.kind`
- `horizon`
- `n_model_scenarios`
- `n_reference_scenarios`
- branch-specific fields inside:
  - `forecast`
  - `unconditional_windowed`
  - `unconditional_path_dataset`

These live in the top-level `protocol` block of the JSON config and are passed into models through the runtime Python contract.

This means a model config should **not** duplicate these values inside `model.params`.

The loader validates this and rejects configs that try to redefine benchmark-owned protocol fields in the model block.

### Why `train_stride` exists

Sequence models often turn a training history into many supervised windows.

If each model extracts those windows differently, the benchmark stops being comparable.

The optional benchmark-level `train_stride` field lets the benchmark define a common training-window extraction stride for sequence models. Models read that value from `train_data.protocol.train_stride` instead of choosing their own hidden default.

For unconditional benchmarks, the benchmark also owns how training paths are constructed:

- `kind = "unconditional_windowed"` means the benchmark cuts one long path into benchmark-owned training paths using `horizon` as the path length and `unconditional_windowed.train_stride`
- `kind = "unconditional_path_dataset"` means the dataset already provides multiple independent training paths, with `unconditional_path_dataset.n_train_paths` training paths and `unconditional_path_dataset.n_realized_paths` held-out realized paths

For the default unconditional benchmark task, path length is fixed:

- `unconditional_windowed`: training path length equals `horizon`
- `unconditional_path_dataset`: training path length and held-out realized path length equal `horizon`

Today `path_dataset` is implemented for synthetic datasets. External CSV/Parquet datasets use `windowed_path`.

In both cases, the runtime normalizes unconditional training into a dataset of paths before handing it to public-contract models.

---

## JSON benchmark contract

A benchmark config controls:

- benchmark identity
- dataset source and dataset-specific parameters
- protocol settings
- metric definitions
- model definitions
- run settings
- output options

### Minimal shape

```json
{
  "version": "1.0",
  "benchmark": {
    "name": "...",
    "dataset": {...},
    "protocol": {...},
    "models": [...]
  }
}
```

### Dataset block

Supported provider kinds:

- `synthetic`
- `csv`
- `parquet`

Synthetic example:

```json
{
  "benchmark": {
    "name": "synthetic_regime_sv",
    "dataset": {
      "name": "synthetic_regime_sv",
      "provider": {
        "kind": "synthetic",
        "config": {
          "generator": "regime_switching_factor_sv",
          "params": {"n_assets": 4, "seed": 11}
        }
      },
      "schema": {
        "layout": "tensor",
        "frequency": "B"
      },
      "semantics": {},
      "metadata": {}
    }
  }
}
```

External CSV example:

```json
{
  "benchmark": {
    "name": "my_returns",
    "dataset": {
      "name": "my_returns",
      "provider": {
        "kind": "csv",
        "config": {
          "path": "../data/my_returns.csv",
          "dropna": "any"
        }
      },
      "schema": {
        "layout": "wide",
        "time_column": "date",
        "target_columns": ["SPX", "SX5E", "NKY"],
        "frequency": "B"
      },
      "semantics": {
        "target_kind": "returns"
      },
      "metadata": {}
    }
  }
}
```

External price-file example:

```json
{
  "benchmark": {
    "name": "my_prices",
    "dataset": {
      "name": "my_prices",
      "provider": {
        "kind": "csv",
        "config": {
          "path": "../data/my_prices.csv"
        }
      },
      "schema": {
        "layout": "wide",
        "time_column": "date",
        "frequency": "B"
      },
      "semantics": {
        "target_kind": "prices",
        "return_kind": "log"
      }
      },
      "metadata": {}
    },
    "protocol": {...},
    "models": [...]
  }
}
```

### Protocol block

The `protocol` block is benchmark-owned and common to all models in the run:

```json
{
  "benchmark": {
    "protocol": {
      "kind": "forecast",
      "horizon": 4,
      "n_model_scenarios": 16,
      "n_reference_scenarios": 32,
      "forecast": {
        "train_size": 180,
        "test_size": 80,
        "context_length": 12,
        "eval_stride": 20,
        "train_stride": 4
      }
    }
  }
}
```

Interpretation:

- `kind`: selects the protocol branch, one of `forecast`, `unconditional_windowed`, or `unconditional_path_dataset`
- `horizon`: number of points each model must generate per evaluation case
- `forecast.train_size`: number of rows used for model fitting in forecast mode
- `forecast.test_size`: number of rows reserved for rolling evaluation in forecast mode
- `forecast.context_length`: length of the conditioning history
- `forecast.eval_stride`: spacing between evaluation origins in the held-out future region
- `forecast.train_stride`: spacing used when the benchmark converts the training region into supervised forecast examples
- `unconditional_windowed.train_size`: number of rows used for unconditional fitting when the benchmark windows one long path
- `unconditional_windowed.test_size`: length of the held-out future region used for unconditional rolling evaluation
- `unconditional_windowed.eval_stride`: spacing between unconditional evaluation origins
- `unconditional_windowed.train_stride`: spacing used when the benchmark cuts unconditional training paths from one long path
- `unconditional_path_dataset.n_train_paths`: number of benchmark-provided training paths
- `unconditional_path_dataset.n_realized_paths`: number of held-out realized paths used at evaluation
- `n_model_scenarios`: number of scenarios requested from each model per evaluation window
- `n_reference_scenarios`: number of reference scenarios drawn when the dataset provides a true generator

### Metrics block

Metrics are configured as metric-definition objects. In config files, the usual pattern is to select built-in metrics by name:

```json
{
  "benchmark": {
    "metrics": [
      {"name": "crps"},
      {"name": "energy_score"},
      {"name": "cross_correlation_error"}
    ]
  }
}
```

If the `metrics` block is omitted, the benchmark uses its built-in default metric set and automatically drops metrics that are not applicable to the current dataset, such as reference-scenario metrics on external CSV/Parquet data.

Each model carries its own preprocessing pipeline definition.

Example:

```json
{
  "pipeline": {
    "name": "standardized",
    "steps": [
      {"type": "standard_scale", "params": {"with_mean": true, "with_std": true}},
      {"type": "clip", "params": {"min_value": -5.0, "max_value": 5.0}}
    ]
  }
}
```

Built-in transforms include:

- `identity`
- `demean`
- `standard_scale`
- `min_max_scale`
- `robust_scale`
- `clip`
- `winsorize`

### Model block

There are three supported model reference kinds:

1. built-in `builtin`
2. discovered external `plugin`
3. direct Python `entrypoint`

Built-in example:

```json
{
  "name": "historical_bootstrap",
  "reference": {
    "kind": "builtin",
    "value": "historical_bootstrap"
  },
  "params": {
    "block_size": 3
  },
  "pipeline": {
    "name": "raw",
    "steps": []
  }
}
```

Plugin example:

```json
{
  "name": "my_research_model",
  "reference": {
    "kind": "plugin",
    "value": "my_research_model"
  },
  "params": {
    "hidden_size": 64,
    "dropout": 0.1
  },
  "pipeline": {
    "name": "raw",
    "steps": []
  }
}
```

Direct entrypoint example:

```json
{
  "name": "entrypoint_model",
  "reference": {
    "kind": "entrypoint",
    "value": "my_package.my_module:build_model"
  },
  "params": {
    "hidden_size": 64
  },
  "pipeline": {
    "name": "raw",
    "steps": []
  }
}
```

Important: do **not** place `kind`, `horizon`, `n_model_scenarios`, `n_reference_scenarios`, or branch-owned protocol fields such as `forecast.context_length`, `forecast.train_size`, `unconditional_windowed.train_stride`, or `unconditional_path_dataset.n_train_paths` inside `model.params`. Those belong to `protocol`.

### Choosing between `builtin`, `plugin`, and `entrypoint`

Use the three model-reference modes for different stages of model maturity:

| Mode | Best for | Config value | Needs packaging/install | Shows up in `ts-benchmark plugins` / UI plugin discovery | Typical owner |
|---|---|---|---|---|---|
| built-in | stable models maintained by this benchmark repo | `reference.kind = "builtin"` | no extra install beyond the benchmark itself | yes | benchmark repo |
| external plugin | models you want to install, share, and discover cleanly across environments | `reference.kind = "plugin"` | yes | yes | external model repo |
| direct entrypoint | local development and rapid iteration against in-progress code | `reference.kind = "entrypoint"` | not necessarily, but the module must be importable | no | external model repo or local workspace |

Practical guidance:

- use `entrypoint` while a model is still under active development and you want the lowest-friction benchmark loop
- use `plugin` once you want the model to be installable, discoverable in the CLI/UI, and runnable by short name across machines
- use `builtin` only for models that are intentionally shipped as part of the benchmark package itself

Recommended workflow:

1. start with `entrypoint` while you are iterating on model code and benchmark compatibility
2. package the model as an external `plugin` once you want clean installation, short-name configs, and CLI/UI discovery
3. promote the model to built-in only if the benchmark repo intends to ship, document, test, and maintain it as part of the benchmark itself

Important behavioral difference:

- `plugin` models appear in `ts-benchmark plugins` and in the Streamlit plugin discovery panel
- `entrypoint` models do **not** appear there; they are loaded only when a config explicitly references their Python import path
- in the Streamlit UI, `entrypoint` models still appear in the "Models declared in current config" panel once they are present in the loaded JSON config

### Why promote a model to built-in

Promoting a model from external `plugin` to built-in is mainly a product and maintenance decision, not a capability requirement.

Advantages of built-in status:

- users can run the model with a short built-in reference value without separately installing a model package
- the benchmark repo can test, document, version, and release that model together with the rest of the framework
- example configs, CLI/UI discovery, and default benchmark workflows work out of the box for all benchmark users
- benchmark maintainers can add benchmark-owned wiring when needed, such as special construction logic or default runtime propagation

Tradeoff:

- once a model is built-in, the benchmark repo effectively owns ongoing compatibility, dependency management, and user support for that model

### What `entrypoint` really means

An `entrypoint` is just a direct Python import path of the form:

```text
package.module:ClassOrFactory
```

This is the recommended development-time path when your model lives outside the benchmark repo and is not yet packaged as a plugin.

Operational implications:

- the Python process running the benchmark must be able to import that module
- this usually means one of:
  - the model repo is installed in the environment
  - the model repo is on `PYTHONPATH`
  - the model code lives directly in the same importable workspace
- because `entrypoint` models are not plugin-discovered, they will not show up in the plugin listing commands or the UI plugin browser, but the Streamlit UI can still show them in the config-model summary for the currently loaded config

### What `plugin` really means

A `plugin` is an installable Python package that:

- registers a model factory through the benchmark's `ts_benchmark.models` entry-point group
- ships a packaged `ts_benchmark_plugin.toml` metadata file next to the builder module

Operational implications:

- the plugin must be installed into the **same Python environment** used to run the benchmark CLI or Streamlit UI
- once installed, the model can be referenced by a short name like `"reference": {"kind": "plugin", "value": "my_model"}`
- the packaged plugin metadata file makes the model discoverable in the CLI/UI and enriches saved run metadata
- if you install or update a plugin while the Streamlit UI is already running, restart the UI process so discovery metadata is refreshed

### Per-model external execution

By default, every model runs in the same Python environment as the benchmark process.

When one model family needs a conflicting dependency stack, you can keep the benchmark in its normal environment and move only that model into a dedicated subprocess/venv with a per-model `execution` block:

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
  "pipeline": {
    "name": "raw",
    "steps": []
  }
}
```

Key points:

- this does **not** change how the model is referenced; only `model.reference` identifies the model
- it only changes *where* that model executes
- the benchmark still passes the same benchmark-owned task and data semantics
- benchmark-level device selection still applies; the selected device is forwarded into the external runner through runtime metadata
- models without an `execution` block still run in-process in the current environment

Use this mode when:

- two model stacks need incompatible package versions
- one model family needs its own venv/container
- you want to keep the benchmark UI/CLI in a stable main environment while isolating a problematic backend

The shipped MXNet examples use this pattern for `gluonts_deepvar` and `gluonts_gpvar`.

### Run block

```json
{
  "run": {
    "seed": 21,
    "execution": {
      "device": "cuda:0",
      "scheduler": "auto"
    },
    "output": {
      "keep_scenarios": true
    }
  }
}
```

The selected device is recorded in benchmark outputs and passed through runtime metadata to models.

---

## Data section

This section is meant to answer:

- what data formats are supported
- how the benchmark interprets a file
- how rolling windows are extracted
- how synthetic and external data differ in the reported metrics

### Supported dataset modes

#### 1. Synthetic datasets

Synthetic datasets are useful for controlled experiments.

The included synthetic baseline is a **regime-switching factor stochastic-volatility generator** designed to reproduce common time-series stylized facts:

- heavy tails
- cross-asset dependence
- volatility clustering
- calm/stress regime changes
- mild leverage-like asymmetry

A simplified form is:

```text
r_{t,i} = μ_{z_t} + β_i σ^m_t ε^m_t + s_i σ^{id}_{t,i} ε^{id}_{t,i}
```

where:

- `z_t` is a latent regime state
- `σ^m_t` is a market-level volatility process
- `σ^{id}_{t,i}` is an idiosyncratic volatility process
- factor and idiosyncratic shocks jointly generate cross-sectional dependence and clustered volatility

Because the synthetic benchmark controls the true conditional data-generating process, it can also draw **reference conditional scenarios** from the same generator. That enables richer distributional metrics beyond realized-path scoring.

#### 2. External datasets

External datasets are for benchmarking on real or user-provided data.

Supported file types:

- CSV
- Parquet

The benchmark can ingest either:

- return series directly, or
- price series which it converts into returns

### Expected tabular layout

The simplest layout is:

- one date column
- one column per asset
- one row per timestamp

Example:

| date | SPX | SX5E | NKY |
|---|---:|---:|---:|
| 2020-01-02 | 0.0071 | 0.0084 | 0.0012 |
| 2020-01-03 | -0.0068 | -0.0091 | -0.0047 |

If the file contains prices instead of returns, set:

```json
"dataset": {
  "semantics": {
    "target_kind": "prices"
  }
}
```

and choose either:

```json
"dataset": {
  "semantics": {
    "return_kind": "simple"
  }
}
```

or:

```json
"dataset": {
  "semantics": {
    "return_kind": "log"
  }
}
```

### Useful dataset fields for tabular providers

- `dataset.provider.config.path`
- `dataset.provider.config.dropna`
- `dataset.provider.config.read_kwargs`
- `dataset.schema.time_column`
- `dataset.schema.target_columns`
- `dataset.semantics.target_kind` (`returns` or `prices`)
- `dataset.semantics.return_kind` (`simple` or `log`)

### How the benchmark slices the data

Given the loaded return matrix, the benchmark:

1. takes the first `train_size` rows as the fit sample
2. takes the next `test_size` rows as the evaluation region
3. rolls forecast origins through the evaluation region using `eval_stride`
4. exposes `context_length` rows before each origin as the conditioning context
5. builds benchmark-owned forecast fit examples whose `history` contains all past rows up to the target origin and whose `context` is the trailing conditioning suffix
6. compares generated `horizon`-step scenarios to the realized future path

This means the benchmark defines the evaluation windows once and all models are tested on the same windows.

### Synthetic vs external metrics

#### Metrics available on all datasets

These compare sampled predictive distributions to realized future paths:

- `crps`
- `energy_score`
- `predictive_mean_mse`
- `coverage_90_error`

#### Metrics available when reference scenarios exist

These are available on synthetic datasets because the benchmark can sample the true conditional future distribution:

- `mean_error`
- `volatility_error`
- `skew_error`
- `excess_kurtosis_error`
- `cross_correlation_error`
- `autocorrelation_error`
- `squared_autocorrelation_error`
- `var_95_error`
- `es_95_error`
- `max_drawdown_error`
- `mmd_rbf`

On external datasets, the benchmark defaults to realized-path scoring metrics because there is no true latent generator available.

### Dataset metadata in outputs

Saved benchmark tables include dataset metadata such as:

- `dataset_name`
- `dataset_source`
- `device`
- `has_reference_scenarios`
- `protocol_kind`
- `generation_mode`
- `path_construction`
- `context_length`
- `horizon`
- `eval_stride`
- `train_stride`
- `n_train_paths`
- `n_realized_paths`

This makes it much easier to compare runs across multiple datasets.

---

## Public model contract

The recommended model-author API is:

```python
from ts_benchmark.model_contract import ...
```

The key objects are:

- `TSGeneratorEstimator`
- `FittedTSGenerator`
- `DataSchema`
- `TSSeries`
- `TrainExample`
- `TrainData`
- `TaskSpec`
- `GenerationRequest`
- `GenerationResult`
- `ModelCapabilities`
- `FitReport`

Important shape conventions:

- `TSSeries.values`: `[time, target_dim]`
- `GenerationResult.samples`: `[num_samples, generated_time, target_dim]`

Important task semantics:

- `task.mode = "forecast"`:
  - `task.horizon` is the forecast horizon
  - `fit(train=...)` receives `TrainData(examples=...)` with `history`, `context`, and `target`
- `task.mode = "unconditional"`:
  - `task.horizon` is the desired generated sequence length
  - `fit(train=...)` receives `TrainData(examples=...)` with `history=None` and `context=None`

Minimal estimator/generator shape:

```python
estimator.fit(train, *, schema, task, valid=None, runtime=None) -> (generator, fit_report)
generator.capabilities() -> ModelCapabilities
generator.sample(request) -> GenerationResult
```

The benchmark also has an internal runtime ABI based on `ScenarioModel`,
`TrainingData`, and `ScenarioRequest`, but that is an internal integration
surface for built-ins, wrappers, and legacy adapters. External model authors
should prefer `ts_benchmark.model_contract`.

---

## Plugin manifests and capability metadata

A packaged plugin manifest is optional but strongly recommended.

The benchmark uses manifests to populate the CLI, the Streamlit UI, and saved run metadata with information such as:

- display name
- model family
- version
- supported dataset sources
- device hints
- whether the model is multivariate
- whether it produces probabilistic samples
- whether it uses the benchmark device setting directly

### Packaged manifest shape

Plugin metadata lives in a packaged `ts_benchmark_plugin.toml` resource. For a single-model package, the top-level shape is:

```toml
[manifest]
display_name = "My research model"
version = "0.1.0"
family = "diffusion"
description = "Research prototype for probabilistic time-series scenarios."
runtime_device_hints = ["cpu", "cuda"]
supported_dataset_sources = ["synthetic", "csv", "parquet"]
required_input = "returns"
default_pipeline = "standardized"
tags = ["research", "diffusion"]

[manifest.capabilities]
multivariate = true
probabilistic_sampling = true
benchmark_protocol_contract = true
explicit_preprocessing = true
uses_benchmark_device = true
```

For multi-model packages, use:

```toml
[plugins.my_model.manifest]
display_name = "My research model"
default_pipeline = "standardized"
```

---

## Extensive how-to for model developers

This section is the intended onboarding path for anyone who wants to benchmark a new model against the models already included.

### Step 1. Keep your model outside the benchmark repo

Recommended structure:

```text
my-model-plugin/
  pyproject.toml
  src/
    my_model_plugin/
      __init__.py
      plugin.py
```

Your model does **not** need to be added under `src/ts_benchmark/model`.

Decision rule:

- if you only want to run benchmarks against your local development code, `entrypoint` is usually enough
- if you want short-name configs, CLI/UI discovery, and a cleaner install story, package the model as an external `plugin`
- only move a model into the benchmark repo itself when you want it maintained as a built-in benchmark model

### Step 2. Implement the public structural contract

Implement the public contract from `ts_benchmark.model_contract`.

```python
from __future__ import annotations

import numpy as np

from ts_benchmark.model_contract import (
    FitReport,
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    ModelCapabilities,
)


class MyGenerator:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supported_modes=frozenset({GenerationMode.FORECAST, GenerationMode.UNCONDITIONAL}),
            supports_multivariate_targets=True,
        )

    def sample(self, request: GenerationRequest) -> GenerationResult:
        values = np.asarray(request.series.values, dtype=float)
        horizon = int(request.task.horizon or 1)
        rng = np.random.default_rng(None if request.runtime is None else request.runtime.seed)
        draws = rng.normal(
            loc=self.mean[None, None, :],
            scale=self.std[None, None, :],
            size=(request.num_samples, horizon, self.mean.shape[0]),
        )
        return GenerationResult(samples=draws)

    def save(self, path):
        raise NotImplementedError


class MyEstimator:
    def fit(self, train, *, schema, task, valid=None, runtime=None):
        del schema, task, valid, runtime
        if getattr(task, "mode", None) == GenerationMode.UNCONDITIONAL:
            values = np.concatenate(
                [np.asarray(example.target.values, dtype=float) for example in train.examples],
                axis=0,
            )
        else:
            values = np.concatenate(
                [
                    np.concatenate(
                        [
                            np.asarray(example.history.values, dtype=float),
                            np.asarray(example.target.values, dtype=float),
                        ],
                        axis=0,
                    )
                    for example in train.examples
                ],
                axis=0,
            )
        x = values.reshape(-1, values.shape[-1])
        generator = MyGenerator(mean=x.mean(axis=0), std=x.std(axis=0, ddof=1) + 1e-6)
        return generator, FitReport(train_metrics={"n_rows": float(x.shape[0])})


def build_estimator(**params):
    del params
    return MyEstimator()
```

If you are writing an in-repo model or an internal wrapper, the older
`ts_benchmark.model` `ScenarioModel` contract still exists, but it is not the
recommended public integration path.

### Step 3. Read task values from the public contract

Use:

```python
mode = task.mode
horizon = task.horizon
num_samples = request.num_samples
history = request.series.values
```

Do **not** make benchmark-owned task settings a second hidden model config.

In particular, do not duplicate:

- generation mode
- horizon
- requested sample count

inside `model.params`.

### Step 4. Make preprocessing assumptions explicit

The benchmark treats preprocessing as part of the experiment definition.

That means:

- your model should expect the benchmark to control preprocessing
- your model should not silently standardize or winsorize data unless that is explicitly part of the benchmark configuration or clearly documented in your plugin manifest / implementation

If your method genuinely requires a specific input normalization, document that in the manifest and in your model documentation.

Use:

- `default_pipeline` for the recommended pipeline
- `required_pipeline` only when the model truly cannot be benchmarked correctly with any other pipeline

### Step 5. Add packaged plugin metadata

Recommended approach:

```toml
[manifest]
display_name = "My research model"
version = "0.1.0"
family = "gaussian"
description = "Example plugin description."
runtime_device_hints = ["cpu", "cuda"]
supported_dataset_sources = ["synthetic", "csv", "parquet"]
required_input = "returns"
default_pipeline = "raw"

[manifest.capabilities]
multivariate = true
probabilistic_sampling = true
benchmark_protocol_contract = true
explicit_preprocessing = true
uses_benchmark_device = true
```

Add `required_pipeline="raw"` only if a different pipeline would make the model
semantically invalid rather than merely suboptimal.

### Step 6. Expose the model as a plugin

In `pyproject.toml`:

```toml
[project.entry-points."ts_benchmark.models"]
my_model = "my_model_plugin.plugin:build_estimator"
```

and in `plugin.py`:

```python
def build_estimator(**params):
    return MyEstimator(**params)
```

This step is what turns your model from "importable Python code" into a discoverable benchmark plugin.

Without this step, you can still benchmark the model through a direct config `reference.kind = "entrypoint"`, but it will not appear in plugin listings or the UI plugin browser.

### Step 7. Add packaged plugin metadata

Add a `ts_benchmark_plugin.toml` file next to your plugin module and include it in package data.

In `pyproject.toml`:

```toml
[tool.setuptools.package-data]
my_model_plugin = ["ts_benchmark_plugin.toml"]
```

Example `ts_benchmark_plugin.toml`:

```toml
[manifest]
display_name = "My model"
default_pipeline = "raw"

[manifest.capabilities]
multivariate = true
probabilistic_sampling = true
```

This is how discoverable plugin metadata is surfaced in the UI and CLI.

### Step 8. Install your plugin in editable mode

```bash
python -m pip install -e /path/to/my-model-plugin
```

Editable install is the recommended workflow while your model is still under development.

Important:

- install the plugin into the same environment where you run `ts-benchmark` or `streamlit run streamlit_app.py`
- if the UI is already running when you install or update the plugin, restart the UI process

### Step 9. Verify discovery

```bash
ts-benchmark plugins
```

or:

```bash
ts-benchmark plugins --json
```

Your model should appear with its manifest and capability metadata.

### Step 10. Create a benchmark config

Reference the plugin by name:

```json
{
  "benchmark": {
    "models": [
      {
        "name": "my_model_run",
        "reference": {
          "kind": "plugin",
          "value": "my_model"
        },
        "params": {
          "hidden_size": 64,
          "dropout": 0.1
        },
        "pipeline": {
          "name": "raw",
          "steps": []
        }
      }
    ]
  }
}
```

If you are still in the pre-plugin phase, the equivalent development-time config is:

```json
{
  "benchmark": {
    "models": [
      {
        "name": "my_model_run",
        "reference": {
          "kind": "entrypoint",
          "value": "my_model_plugin.plugin:build_model"
        },
        "params": {
          "hidden_size": 64,
          "dropout": 0.1
        },
        "pipeline": {
          "name": "raw",
          "steps": []
        }
      }
    ]
  }
}
```

That approach is often the fastest way to test a new model against the benchmark before you decide whether packaging it as a plugin is worth the extra ceremony.

Remember:

- model hyperparameters live in `model.params`
- benchmark protocol values live in `benchmark.protocol`
- preprocessing lives in `model.pipeline`

### Step 11. Run the benchmark

```bash
ts-benchmark run my_config.json
```

Outputs include:

- metrics
- ranks
- config snapshot
- model infos
- plugin manifest metadata
- summary metadata
- optionally saved scenarios

### Step 12. Compare fairly

For fair comparisons, keep these choices fixed across models unless the whole experiment is explicitly about varying them:

- dataset
- `protocol.kind`
- `horizon`
- forecast branch fields such as `forecast.train_size`, `forecast.test_size`, `forecast.context_length`, `forecast.eval_stride`, `forecast.train_stride`
- unconditional branch fields such as `unconditional_windowed.train_size`, `unconditional_windowed.test_size`, `unconditional_windowed.eval_stride`, `unconditional_windowed.train_stride`, `unconditional_path_dataset.n_train_paths`, `unconditional_path_dataset.n_realized_paths`
- scenario counts
- preprocessing pipeline
- benchmark seed

### Step 13. Understand how device selection works

The benchmark has a benchmark-level runtime device field.

Model authors should read the runtime device through `train_data.runtime.device` and `request.runtime.device` if their implementation can use it.

If a model ignores device selection, document that in the manifest by setting `uses_benchmark_device=False`.

Current runtime behavior:

- `device: null` or UI `auto` means:
  - use all visible CUDA GPUs if available
  - otherwise use `mps` if available
  - otherwise fall back to `cpu`
- `device: "cuda:0"` pins the run to one specific GPU
- `device: "cuda:0,cuda:1"` restricts the run to a specific GPU subset

When more than one CUDA device is available and the benchmark contains more than one model, the benchmark schedules models round-robin across the selected GPUs in separate worker processes.

If a model is configured with a per-model external `execution` block, that device assignment is still forwarded into the child process. The child model may still fall back internally if its backend cannot actually honor that device.

### Step 14. Provide useful `model_info()` metadata

A model can optionally implement:

```python
def model_info(self) -> dict:
    ...
```

This information is saved in run artifacts and is useful for debugging and reporting resolved configuration details.

Typical things to include:

- architecture sizes
- diffusion steps
- backend name
- resolved device
- training loss summary
- any derived internal settings that help users interpret the run

### Step 15. Use the included example plugin package

This repository includes a complete external plugin example under:

```text
plugin_examples/eqbench_demo_gaussian_plugin/
```

and an example config under:

```text
plugin_examples/demo_gaussian_config.json
```

That example now includes both a model entry point and a manifest entry point.

---

## Built-in models currently included

### Historical bootstrap

Resamples historical return vectors from the training set.

Strengths:

- preserves empirical marginal behavior
- preserves same-date cross-sectional dependence
- very simple baseline

Limitation:

- does not explicitly model dynamic volatility

### Historical bootstrap with stochastic volatility

Uses EWMA volatility scaling, bootstraps standardized residuals, and simulates future volatility before re-inflating residuals.

---

## CLI and UI

### CLI

Validate:

```bash
ts-benchmark validate smoke_test
```

Run:

```bash
ts-benchmark run synthetic_basic_benchmark
```

List plugins:

```bash
ts-benchmark plugins
```

List plugins as structured JSON:

```bash
ts-benchmark plugins --json
```

### Streamlit UI

The UI lets you:

- load an example config
- upload a config JSON
- choose a bundled dataset or upload an external dataset
- select an execution device
- inspect discovered model plugins and their manifests
- inspect models declared in the current config, including direct `entrypoint` models
- run the benchmark and inspect metrics, ranks, dataset summary, and saved model metadata
- browse MLflow experiments, runs, and logged benchmark artifacts from a tracking URI when `mlflow` is installed

Launch it with:

```bash
streamlit run streamlit_app.py
```

---

## Optional MLflow tracking

Benchmark runs can be logged to MLflow through an optional `run.tracking.mlflow` block.

Example:

```json
{
  "run": {
    "execution": {
      "scheduler": "auto"
    },
    "output": {},
    "tracking": {
      "mlflow": {
        "enabled": true,
        "tracking_uri": "sqlite:///mlflow.db",
        "experiment_name": "ts-benchmark-dev",
        "run_name": "constant-vol-smoke",
        "tags": {
          "owner": "research"
        },
        "log_artifacts": true,
        "log_model_info": true,
        "log_diagnostics": true,
        "log_scenarios": false
      },
    }
  }
}
```

When enabled, the benchmark logs:

- flattened benchmark/protocol/model parameters
- per-model metrics and average ranks
- functional smoke pass/fail summary when diagnostics are enabled
- benchmark artifacts such as `metrics.csv`, `ranks.csv`, config JSON, summary JSON, model info, and optional diagnostics/scenarios

Tracking stays optional:

- if `run.tracking.mlflow.enabled` is `false`, the benchmark behaves exactly as before
- subprocess model workers do not create their own MLflow runs; only the parent benchmark run is logged

For new setups, prefer a database-backed tracking URI such as `sqlite:///mlflow.db`. MLflow's filesystem backend still works, but current MLflow versions warn that it is deprecated.

---

## Outputs

A typical run can save:

- `metrics.csv`
- `ranks.csv`
- `benchmark_config.json`
- `run.json`
- `model_results.json`
- `summary.json`
- `scenarios.npz` (optional)

### What `model_results.json` now contains

For each model result, the saved metadata includes:

- model config reference object (`kind` and `value`)
- model params
- pipeline summary
- execution info
- declared plugin manifest
- fitted model info, when provided by the model
- runtime-discovered manifest, when available
- metric results and ranks
- scenario output shape when scenarios were kept

### What the benchmark results always mention

At minimum, run metadata includes:

- dataset name
- dataset source
- selected device
- whether reference scenarios exist
- `protocol_kind`
- `context_length`
- `horizon`
- `eval_stride`
- `train_stride`
- `path_construction`
- `n_train_paths`

---

## Example files

Useful starting points:

- `smoke_test`
- `synthetic_basic_benchmark`
- `plugin_examples/demo_gaussian_config.json`

---

## Notes

- Synthetic data is only the starting point; the benchmark is designed to support real external datasets as first-class inputs.
- The benchmark contract is intentionally small so that external researchers can add new models with minimal friction.
- The plugin manifest layer is descriptive and developer-facing: it improves discoverability without forcing model code to live inside the benchmark repository.
