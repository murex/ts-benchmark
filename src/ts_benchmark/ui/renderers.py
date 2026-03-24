"""Structured renderers for the Streamlit UI."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


def _safe_json(obj: object) -> str:
    def _default(value: object):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return str(value)

    return json.dumps(obj, indent=2, default=_default)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _display_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    return str(value)


def _list_kind(values: list[Any]) -> str:
    if not values:
        return "scalar"
    if all(isinstance(item, dict) for item in values):
        keys = {tuple(sorted(item.keys())) for item in values}
        return "dict" if len(keys) == 1 else "complex"
    if all(not isinstance(item, (dict, list, tuple)) for item in values):
        return "scalar"
    if all(isinstance(item, (list, tuple)) for item in values):
        widths = {len(item) for item in values}
        return "rows" if len(widths) == 1 else "complex"
    return "complex"


def _normalize_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = df.replace({np.nan: None})
    records: list[dict[str, Any]] = []
    for _, row in normalized.iterrows():
        records.append({str(column): row[column] for column in normalized.columns})
    return records


def _primitive_input(label: str, value: Any, *, key: str) -> Any:
    if isinstance(value, bool):
        return st.checkbox(label, value=value, key=key)
    if isinstance(value, int) and not isinstance(value, bool):
        return st.number_input(label, value=int(value), step=1, key=key)
    if isinstance(value, float):
        return st.number_input(label, value=float(value), key=key)
    if value is None:
        return st.text_input(label, value="", key=key)
    return st.text_input(label, value=str(value), key=key)


def render_status_badge(status: str) -> None:
    normalized = str(status).lower()
    if normalized in {"pass", "valid", "success", "succeeded"}:
        st.success(status)
    elif normalized in {"running", "active"}:
        st.info(status)
    elif normalized in {"warning", "warn"}:
        st.warning(status)
    else:
        st.error(status)


def render_cli_command(command: str) -> None:
    st.caption("Equivalent CLI")
    st.code(command, language="bash")


def render_json_advanced(value: object, *, label: str) -> None:
    with st.expander(f"{label} (Advanced JSON)", expanded=False):
        st.code(_safe_json(value), language="json")


def render_scalar_list(values: list[Any], *, editable: bool = False, key: str = "") -> list[Any]:
    frame = pd.DataFrame({"value": list(values)})
    if editable:
        edited = st.data_editor(frame, num_rows="dynamic", use_container_width=True, key=key or None)
        return edited["value"].tolist()
    st.dataframe(frame, use_container_width=True, hide_index=True)
    return list(values)


def render_table(records: list[dict[str, Any]] | list[list[Any]], *, editable: bool = False, key: str = "") -> Any:
    if records and isinstance(records[0], dict):
        frame = pd.DataFrame(records)
    else:
        frame = pd.DataFrame(records)
    if editable:
        edited = st.data_editor(frame, num_rows="dynamic", use_container_width=True, key=key or None)
        if records and isinstance(records[0], dict):
            return _normalize_records(edited)
        return edited.replace({np.nan: None}).values.tolist()
    st.dataframe(frame, use_container_width=True, hide_index=True)
    return records


def render_key_value(
    obj: dict[str, Any],
    *,
    editable: bool = False,
    key_prefix: str = "",
    allow_add_fields: bool = True,
) -> dict[str, Any]:
    updated = dict(obj)
    primitive_rows: list[dict[str, Any]] = []

    for key, value in list(updated.items()):
        widget_key = f"{key_prefix}.{key}" if key_prefix else str(key)
        if _is_scalar(value):
            if editable:
                updated[key] = _primitive_input(str(key), value, key=widget_key)
            else:
                primitive_rows.append({"key": str(key), "value": _display_scalar(value)})
            continue

        with st.expander(str(key), expanded=False):
            if isinstance(value, dict):
                updated[key] = render_key_value(
                    value,
                    editable=editable,
                    key_prefix=widget_key,
                    allow_add_fields=allow_add_fields,
                )
            elif isinstance(value, list):
                updated[key] = render_structured_value(
                    value,
                    label=str(key),
                    editable=editable,
                    key_prefix=widget_key,
                    allow_add_fields=allow_add_fields,
                )
            else:
                st.write(value)

    if primitive_rows:
        st.dataframe(pd.DataFrame(primitive_rows), use_container_width=True, hide_index=True)

    if editable and allow_add_fields:
        add_prefix = f"{key_prefix}.add" if key_prefix else "add"
        with st.expander("Add field", expanded=False):
            new_key = st.text_input("Field name", value="", key=f"{add_prefix}.name")
            value_type = st.selectbox(
                "Field type",
                options=["string", "int", "float", "bool"],
                key=f"{add_prefix}.type",
            )
            if value_type == "bool":
                new_value = st.checkbox("Field value", value=False, key=f"{add_prefix}.bool")
            elif value_type == "int":
                new_value = st.number_input("Field value", value=0, step=1, key=f"{add_prefix}.int")
            elif value_type == "float":
                new_value = st.number_input("Field value", value=0.0, key=f"{add_prefix}.float")
            else:
                new_value = st.text_input("Field value", value="", key=f"{add_prefix}.string")
            if st.button("Add field", key=f"{add_prefix}.submit") and new_key:
                updated[new_key] = new_value

    return updated


def render_structured_value(
    value: Any,
    *,
    label: str,
    editable: bool = False,
    key_prefix: str = "",
    allow_add_fields: bool = True,
) -> Any:
    if isinstance(value, dict):
        return render_key_value(
            value,
            editable=editable,
            key_prefix=key_prefix or label,
            allow_add_fields=allow_add_fields,
        )

    if isinstance(value, list):
        kind = _list_kind(value)
        if kind == "scalar":
            return render_scalar_list(value, editable=editable, key=key_prefix or label)
        if kind in {"dict", "rows"}:
            return render_table(value, editable=editable, key=key_prefix or label)
        st.caption(label)
        render_json_advanced(value, label=label)
        return value

    if isinstance(value, pd.DataFrame):
        st.dataframe(value, use_container_width=True, hide_index=True)
        return value

    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim == 1:
            st.dataframe(pd.DataFrame({"value": array}), use_container_width=True, hide_index=True)
        else:
            st.dataframe(pd.DataFrame(array), use_container_width=True, hide_index=True)
        return value

    st.write(value)
    return value
