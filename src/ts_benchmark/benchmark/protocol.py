"""Shared benchmark protocol definition used in config and runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    train_size: int
    test_size: int
    context_length: int
    horizon: int
    generation_mode: str = "forecast"
    eval_stride: int = 1
    train_stride: int = 1
    n_model_scenarios: int = 64
    n_reference_scenarios: int = 128

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for field_name in ("train_size", "test_size", "context_length", "horizon",
                           "eval_stride", "train_stride", "n_model_scenarios",
                           "n_reference_scenarios"):
            value = getattr(self, field_name)
            if not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}.")
        if not isinstance(self.generation_mode, str):
            raise TypeError(
                f"generation_mode must be a string, got {type(self.generation_mode).__name__}."
            )
        if self.generation_mode not in {"forecast", "unconditional"}:
            raise ValueError(
                "generation_mode must be either 'forecast' or 'unconditional'."
            )
        if self.train_size < 2:
            raise ValueError("train_size must be at least 2.")
        if self.test_size < 1:
            raise ValueError("test_size must be positive.")
        if self.generation_mode == "forecast":
            if self.context_length < 1:
                raise ValueError("context_length must be positive for forecast mode.")
        elif self.context_length != 0:
            raise ValueError("context_length must be zero for unconditional mode.")
        if self.horizon < 1:
            raise ValueError("horizon must be positive.")
        if self.eval_stride < 1:
            raise ValueError("eval_stride must be positive.")
        if self.train_stride < 1:
            raise ValueError("train_stride must be positive.")
        if self.n_model_scenarios < 1:
            raise ValueError("n_model_scenarios must be positive.")
        if self.n_reference_scenarios < 1:
            raise ValueError("n_reference_scenarios must be positive.")
