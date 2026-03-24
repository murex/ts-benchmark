"""Session-state helpers for the Streamlit UI."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import streamlit as st

PAGE_KEY = "ui.page"
CURRENT_CONFIG_KEY = "cfg.current"
CURRENT_CONFIG_PATH_KEY = "cfg.path"
EFFECTIVE_CONFIG_KEY = "cfg.effective"
VALIDATION_KEY = "cfg.validation"
CURRENT_RUN_KEY = "run.current"
CURRENT_RUN_ARTIFACTS_KEY = "run.current_artifacts"
SELECTED_RUN_DIR_KEY = "results.selected_run_dir"
COMPARE_RUN_DIR_KEY = "results.compare_run_dir"
TRACKING_URI_KEY = "tracking.uri"
TRACKING_EXPERIMENTS_KEY = "tracking.selected_experiment_ids"
TRACKING_RUN_KEY = "tracking.selected_run_id"


def _deepcopy_default(value: Any) -> Any:
    return copy.deepcopy(value)


def init_state() -> None:
    defaults = {
        PAGE_KEY: "Home",
        CURRENT_CONFIG_KEY: {},
        CURRENT_CONFIG_PATH_KEY: None,
        EFFECTIVE_CONFIG_KEY: {},
        VALIDATION_KEY: None,
        CURRENT_RUN_KEY: None,
        CURRENT_RUN_ARTIFACTS_KEY: None,
        SELECTED_RUN_DIR_KEY: None,
        COMPARE_RUN_DIR_KEY: None,
        TRACKING_URI_KEY: None,
        TRACKING_EXPERIMENTS_KEY: [],
        TRACKING_RUN_KEY: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = _deepcopy_default(value)


def get_current_config() -> dict[str, Any]:
    return copy.deepcopy(st.session_state.get(CURRENT_CONFIG_KEY, {}))


def set_current_config(config: dict[str, Any]) -> None:
    st.session_state[CURRENT_CONFIG_KEY] = copy.deepcopy(config)


def get_current_config_path() -> Path | None:
    raw = st.session_state.get(CURRENT_CONFIG_PATH_KEY)
    return None if raw is None else Path(raw)


def set_current_config_path(path: str | Path | None) -> None:
    st.session_state[CURRENT_CONFIG_PATH_KEY] = None if path is None else str(Path(path))


def get_effective_config() -> dict[str, Any]:
    return copy.deepcopy(st.session_state.get(EFFECTIVE_CONFIG_KEY, {}))


def set_effective_config(config: dict[str, Any]) -> None:
    st.session_state[EFFECTIVE_CONFIG_KEY] = copy.deepcopy(config)


def get_validation_result() -> dict[str, Any] | None:
    value = st.session_state.get(VALIDATION_KEY)
    return None if value is None else copy.deepcopy(value)


def set_validation_result(result: dict[str, Any] | None) -> None:
    st.session_state[VALIDATION_KEY] = None if result is None else copy.deepcopy(result)


def get_current_run() -> dict[str, Any] | None:
    return st.session_state.get(CURRENT_RUN_KEY)


def set_current_run(run_info: dict[str, Any] | None) -> None:
    st.session_state[CURRENT_RUN_KEY] = run_info


def get_current_run_artifacts() -> dict[str, Any] | None:
    value = st.session_state.get(CURRENT_RUN_ARTIFACTS_KEY)
    return None if value is None else copy.deepcopy(value)


def set_current_run_artifacts(artifacts: dict[str, Any] | None) -> None:
    st.session_state[CURRENT_RUN_ARTIFACTS_KEY] = None if artifacts is None else copy.deepcopy(artifacts)


def get_selected_run_dir() -> Path | None:
    raw = st.session_state.get(SELECTED_RUN_DIR_KEY)
    return None if raw is None else Path(raw)


def set_selected_run_dir(path: str | Path | None) -> None:
    st.session_state[SELECTED_RUN_DIR_KEY] = None if path is None else str(Path(path))


def get_compare_run_dir() -> Path | None:
    raw = st.session_state.get(COMPARE_RUN_DIR_KEY)
    return None if raw is None else Path(raw)


def set_compare_run_dir(path: str | Path | None) -> None:
    st.session_state[COMPARE_RUN_DIR_KEY] = None if path is None else str(Path(path))


def get_page() -> str:
    return str(st.session_state.get(PAGE_KEY, "Home"))


def set_page(page: str) -> None:
    st.session_state[PAGE_KEY] = str(page)

