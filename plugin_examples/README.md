# Plugin example

This directory contains an example model plugin package that lives **outside** the benchmark source tree.

## Install the benchmark core first

From the benchmark project root:

```bash
python -m pip install -e .
```

## Install the example plugin in editable mode

```bash
python -m pip install -e plugin_examples/eqbench_demo_gaussian_plugin
```

## Verify discovery

```bash
ts-benchmark plugins
```

After installation, the plugin name `demo_gaussian_plugin` becomes available to benchmark configs through `model.reference`, for example `{"kind": "plugin", "value": "demo_gaussian_plugin"}`.

This example also ships a packaged `ts_benchmark_plugin.toml` resource so that the benchmark UI and CLI can show capabilities such as supported dataset sources, probabilistic sampling support, and device hints without any extra Python-side manifest entry point.
