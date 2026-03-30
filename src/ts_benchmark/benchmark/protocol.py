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
    unconditional_train_data_mode: str | None = None
    unconditional_train_window_length: int | None = None
    unconditional_n_train_paths: int | None = None
    n_model_scenarios: int = 64
    n_reference_scenarios: int = 128

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for field_name in (
            "train_size",
            "test_size",
            "context_length",
            "horizon",
            "eval_stride",
            "train_stride",
            "n_model_scenarios",
            "n_reference_scenarios",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}.")
        if self.unconditional_train_window_length is not None and not isinstance(
            self.unconditional_train_window_length, int
        ):
            raise TypeError(
                "unconditional_train_window_length must be an integer or None, "
                f"got {type(self.unconditional_train_window_length).__name__}."
            )
        if self.unconditional_train_data_mode is not None and not isinstance(
            self.unconditional_train_data_mode, str
        ):
            raise TypeError(
                "unconditional_train_data_mode must be a string or None, "
                f"got {type(self.unconditional_train_data_mode).__name__}."
            )
        if self.unconditional_n_train_paths is not None and not isinstance(
            self.unconditional_n_train_paths, int
        ):
            raise TypeError(
                "unconditional_n_train_paths must be an integer or None, "
                f"got {type(self.unconditional_n_train_paths).__name__}."
            )
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
            if self.unconditional_train_data_mode is not None:
                raise ValueError(
                    "unconditional_train_data_mode must be omitted in forecast mode."
                )
            if self.unconditional_train_window_length is not None:
                raise ValueError(
                    "unconditional_train_window_length must be omitted in forecast mode."
                )
            if self.unconditional_n_train_paths is not None:
                raise ValueError(
                    "unconditional_n_train_paths must be omitted in forecast mode."
                )
        elif self.context_length != 0:
            raise ValueError("context_length must be zero for unconditional mode.")
        elif self.unconditional_train_data_mode not in {"windowed_path", "path_dataset"}:
            raise ValueError(
                "unconditional_train_data_mode must be either 'windowed_path' or 'path_dataset' "
                "for unconditional mode."
            )
        elif self.unconditional_train_data_mode == "windowed_path":
            if self.unconditional_train_window_length is None:
                raise ValueError(
                    "unconditional_train_window_length is required when "
                    "unconditional_train_data_mode='windowed_path'."
                )
            if self.unconditional_train_window_length < 1:
                raise ValueError("unconditional_train_window_length must be positive.")
            if self.unconditional_train_window_length > self.train_size:
                raise ValueError(
                    "unconditional_train_window_length must be less than or equal to train_size."
                )
            if self.unconditional_n_train_paths is not None:
                raise ValueError(
                    "unconditional_n_train_paths must be omitted when "
                    "unconditional_train_data_mode='windowed_path'."
                )
        else:
            if self.unconditional_n_train_paths is None:
                raise ValueError(
                    "unconditional_n_train_paths is required when "
                    "unconditional_train_data_mode='path_dataset'."
                )
            if self.unconditional_n_train_paths < 1:
                raise ValueError("unconditional_n_train_paths must be positive.")
            if self.unconditional_train_window_length is not None:
                raise ValueError(
                    "unconditional_train_window_length must be omitted when "
                    "unconditional_train_data_mode='path_dataset'."
                )
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
