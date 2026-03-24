"""Environment page for the Streamlit UI."""

from __future__ import annotations

import streamlit as st

from ..renderers import render_structured_value
from ..services.environment import discover_plugins_df, environment_summary, inspect_subprocess_envs
from ..state import get_current_config


def render() -> None:
    st.header("Environment")
    st.caption("Inspect local runtime details, discovered plugins, device availability, and subprocess runner configuration.")

    render_structured_value(environment_summary(), label="environment", editable=False, key_prefix="environment.summary")

    st.subheader("Discovered plugins")
    plugins = discover_plugins_df()
    plugin_error = plugins.attrs.get("error")
    if plugin_error:
        st.error(f"Plugin discovery failed: {plugin_error}")
    if plugins.empty:
        st.info("No plugins discovered.")
    else:
        st.dataframe(plugins, use_container_width=True, hide_index=True)

    st.subheader("Subprocess execution plan")
    config = get_current_config()
    if not config:
        st.info("No config is currently loaded.")
        return
    execution = inspect_subprocess_envs(config)
    if execution.empty:
        st.info("No models are currently declared in the config.")
    else:
        st.dataframe(execution, use_container_width=True, hide_index=True)
