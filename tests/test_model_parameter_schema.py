from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.model.catalog.plugins import (
    ModelParameterSchema,
    ModelParameterSpec,
    extract_model_parameter_schema,
)


def test_extract_model_parameter_schema_prefers_formal_builder_schema() -> None:
    def build_model(**params):
        return params

    build_model.PARAMETER_SCHEMA = ModelParameterSchema(
        name="formal_plugin",
        fields=(
            ModelParameterSpec(
                name="ridge",
                value_type="float",
                default=1e-6,
                annotation="float",
                schema_source="builder_attribute",
            ),
        ),
        schema_source="builder_attribute",
    )

    schema = extract_model_parameter_schema(build_model, default_name="formal_plugin")

    assert schema is not None
    assert schema.schema_source == "builder_attribute"
    assert [(field.name, field.value_type) for field in schema.fields] == [("ridge", "float")]


def test_extract_model_parameter_schema_infers_builder_wrapping_model_class() -> None:
    class DemoPluginModel:
        def __init__(self, ridge: float = 1e-6, enabled: bool = True):
            self.ridge = ridge
            self.enabled = enabled

    def build_model(**params):
        return DemoPluginModel(**params)

    schema = extract_model_parameter_schema(build_model, default_name="demo_plugin")

    assert schema is not None
    assert schema.schema_source == "builder_referenced_class"
    assert [(field.name, field.value_type) for field in schema.fields] == [
        ("ridge", "float"),
        ("enabled", "bool"),
    ]


def test_extract_model_parameter_schema_infers_builder_wrapping_dataclass_config() -> None:
    @dataclass
    class DemoConfig:
        hidden_size: int = 64
        dropout_rate: float = 0.1
        lags_seq: tuple[int, ...] = (1, 5)
        trainer_kwargs: dict[str, int] = field(default_factory=lambda: {"epochs": 5})

    class DemoAdapter:
        def __init__(self, config: DemoConfig):
            self.config = config

    def build_model(**params):
        return DemoAdapter(DemoConfig(**params))

    schema = extract_model_parameter_schema(build_model, default_name="demo_adapter")

    assert schema is not None
    assert schema.schema_source == "builder_referenced_dataclass"
    assert [(field.name, field.value_type) for field in schema.fields] == [
        ("hidden_size", "int"),
        ("dropout_rate", "float"),
        ("lags_seq", "list"),
        ("trainer_kwargs", "dict"),
    ]
