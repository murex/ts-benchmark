"""Run and diagnostics definition objects."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model.definition import ModelExecutionConfig
from ..utils import JsonObject, StringMap


@dataclass
class FunctionalSmokeConfig:
    enabled: bool = False
    finite_required: bool = True
    mean_abs_error_max: float | None = None
    std_ratio_min: float | None = None
    std_ratio_max: float | None = None
    crps_max: float | None = None
    energy_score_max: float | None = None
    cross_correlation_error_max: float | None = None


@dataclass
class DiagnosticsConfig:
    save_model_debug_artifacts: bool = False
    save_distribution_summary: bool = False
    save_per_window_metrics: bool = False
    functional_smoke: FunctionalSmokeConfig = field(default_factory=FunctionalSmokeConfig)

    def enabled(self) -> bool:
        return bool(
            self.save_model_debug_artifacts
            or self.save_distribution_summary
            or self.save_per_window_metrics
            or self.functional_smoke.enabled
        )


@dataclass
class MlflowTrackingConfig:
    enabled: bool = False
    tracking_uri: str | None = None
    experiment_name: str = "ts-benchmark"
    run_name: str | None = None
    tags: StringMap = field(default_factory=StringMap)
    log_artifacts: bool = True
    log_model_info: bool = True
    log_diagnostics: bool = True
    log_scenarios: bool = False



@dataclass
class TrackingConfig:
    mlflow: MlflowTrackingConfig = field(default_factory=MlflowTrackingConfig)

    def enabled(self) -> bool:
        return bool(self.mlflow.enabled)



@dataclass
class OutputConfig:
    output_dir: str | None = None
    keep_scenarios: bool = False
    save_scenarios: bool = False
    save_model_info: bool = True
    save_summary: bool = True



@dataclass
class RunConfig:
    name: str | None = None
    description: str | None = None
    seed: int = 7
    device: str | None = None
    scheduler: str = "auto"
    model_execution: ModelExecutionConfig = field(default_factory=ModelExecutionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metadata: JsonObject = field(default_factory=JsonObject)
