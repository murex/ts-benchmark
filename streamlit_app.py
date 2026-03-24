"""Streamlit shell for the TS benchmark UI."""

from __future__ import annotations

import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from ts_benchmark.ui.pages import (
    config_studio,
    data_studio,
    home,
    model_catalog,
    results,
    run_lab,
    tracking,
)
from ts_benchmark.ui.services.configs import default_config_dict
from ts_benchmark.ui.state import get_current_config, get_page, init_state, set_current_config, set_page

PAGES = {
    "Home": home.render,
    "Data Studio": data_studio.render,
    "Model Catalog": model_catalog.render,
    "Benchmarks": config_studio.render,
    "Run Lab": run_lab.render,
    "Results Explorer": results.render,
    "Experiment Tracking": tracking.render,
}


def init_app() -> None:
    st.set_page_config(page_title="TS Benchmark", layout="wide")
    init_state()
    if not get_current_config():
        set_current_config(default_config_dict())


def build_navigation() -> str:
    current_page = get_page()
    page_names = list(PAGES)
    if current_page not in page_names:
        current_page = page_names[0]
        set_page(current_page)

    with st.sidebar:
        st.title("TS Benchmark")
        st.caption("Workspace")
        selected = st.radio(
            "Navigation",
            options=page_names,
            index=page_names.index(current_page),
            label_visibility="collapsed",
        )
        if selected != current_page:
            set_page(selected)
            current_page = selected

    return current_page


def dispatch_page(page_name: str) -> None:
    render = PAGES[page_name]
    render()


def main() -> None:
    init_app()
    page_name = build_navigation()
    dispatch_page(page_name)


if __name__ == "__main__":
    main()
