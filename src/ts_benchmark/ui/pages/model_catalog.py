"""Model catalog page for the Streamlit UI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ..services.environment import discover_plugins_df
from ..services.model_catalog import (
    build_file_entrypoint_value,
    describe_catalog_model,
    describe_catalog_model_entry,
    delete_catalog_model,
    find_repo_scenario_model_candidates,
    inspect_entrypoint_python_file,
    list_entrypoint_python_files,
    list_model_catalog,
    save_catalog_model,
    store_uploaded_entrypoint_file,
)

MODEL_CATALOG_SELECTED_KEY = "model_catalog.selected"
MODEL_CATALOG_PENDING_SELECTED_KEY = "model_catalog.pending_selected"
MODEL_CATALOG_FLASH_KEY = "model_catalog.flash"
MODEL_CATALOG_PLUGIN_NAME_KEY = "model_catalog.add.plugin_name"
MODEL_CATALOG_PLUGIN_NAME_DEFAULT_KEY = "model_catalog.add.plugin_name_default"
MODEL_CATALOG_ENTRYPOINT_NAME_KEY = "model_catalog.add.entrypoint_name"
MODEL_CATALOG_ENTRYPOINT_NAME_DEFAULT_KEY = "model_catalog.add.entrypoint_name_default"
MODEL_CATALOG_ENTRYPOINT_UPLOAD_MARKER_KEY = "model_catalog.add.entrypoint_upload_marker"
MODEL_CATALOG_ENTRYPOINT_UPLOADED_PATH_KEY = "model_catalog.add.entrypoint_uploaded_path"
MODEL_CATALOG_ENTRYPOINT_REPO_ROOT_KEY = "model_catalog.add.entrypoint_repo_root"
MODEL_CATALOG_ENTRYPOINT_FILE_KEY = "model_catalog.add.entrypoint_file_select"


def _set_flash(level: str, message: str) -> None:
    st.session_state[MODEL_CATALOG_FLASH_KEY] = {
        "level": str(level),
        "message": str(message),
    }


def _render_flash() -> None:
    payload = st.session_state.pop(MODEL_CATALOG_FLASH_KEY, None)
    if not payload:
        return
    level = payload.get("level")
    message = payload.get("message")
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)


def _entrypoint_default_name(entrypoint: str) -> str:
    text = str(entrypoint or "").strip()
    if not text:
        return ""
    _, _, attr = text.partition(":")
    target = attr or text.rsplit(".", maxsplit=1)[-1]
    return target.rsplit(".", maxsplit=1)[-1]


def _uploaded_file_signature(uploaded: object) -> str:
    return ":".join(
        [
            str(getattr(uploaded, "name", "")),
            str(getattr(uploaded, "size", "")),
            str(getattr(uploaded, "file_id", "")),
        ]
    )


def _handle_uploaded_entrypoint_file(uploaded: object | None) -> str | None:
    if uploaded is None:
        return None

    signature = _uploaded_file_signature(uploaded)
    if st.session_state.get(MODEL_CATALOG_ENTRYPOINT_UPLOAD_MARKER_KEY) != signature:
        stored_path = store_uploaded_entrypoint_file(
            filename=str(getattr(uploaded, "name", "")),
            content=uploaded.getvalue(),
        )
        st.session_state[MODEL_CATALOG_ENTRYPOINT_UPLOADED_PATH_KEY] = str(stored_path)
        st.session_state[MODEL_CATALOG_ENTRYPOINT_UPLOAD_MARKER_KEY] = signature
        st.session_state.pop("model_catalog.add.entrypoint_symbol", None)

    return str(st.session_state.get(MODEL_CATALOG_ENTRYPOINT_UPLOADED_PATH_KEY) or "")


def _format_parameter_default(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return str(value)
    return str(value)


def _display_path(path: str) -> str:
    candidate = Path(path)
    cwd = Path.cwd()
    return str(candidate.relative_to(cwd)) if candidate.is_relative_to(cwd) else str(candidate)


def _simplified_parameter_rows(parameters: list[dict[str, object]]) -> tuple[list[dict[str, object]], bool]:
    explicit_rows: list[dict[str, object]] = []
    accepts_varargs = False
    for parameter in parameters:
        if str(parameter.get("parameter_type") or "") == "vararg":
            accepts_varargs = True
            continue
        explicit_rows.append(
            {
                "parameter": parameter.get("name"),
                "type": parameter.get("value_type") or parameter.get("annotation") or "",
                "default": _format_parameter_default(parameter.get("default")),
                "required": bool(parameter.get("required", False)),
            }
        )
    return explicit_rows, accepts_varargs


def _load_detail_rows(entry: dict[str, object], resolution: dict[str, object]) -> list[dict[str, str]]:
    rows = [
        {"field": "Reference kind", "value": str(((entry.get("reference") or {}).get("kind") or "")).strip()},
        {"field": "Reference", "value": str(((entry.get("reference") or {}).get("value") or "")).strip()},
    ]
    status = str(resolution.get("status") or "").strip()
    if status:
        rows.append({"field": "Status", "value": status})
    source = str(resolution.get("source") or "").strip()
    if source:
        rows.append({"field": "Source", "value": source})
    target = str(resolution.get("target") or "").strip()
    reference_value = str(((entry.get("reference") or {}).get("value") or "")).strip()
    if target and target != reference_value:
        rows.append({"field": "Target", "value": target})
    package = str(resolution.get("package") or "").strip()
    if package:
        rows.append({"field": "Package", "value": package})
    version = str(resolution.get("version") or "").strip()
    if version:
        rows.append({"field": "Version", "value": version})
    error = str(resolution.get("error") or "").strip()
    if error:
        rows.append({"field": "Error", "value": error})
    return [row for row in rows if row["value"]]


def _sync_name_default(name_key: str, marker_key: str, default_value: str) -> None:
    previous_default = str(st.session_state.get(marker_key) or "")
    current_value = str(st.session_state.get(name_key) or "")
    if name_key not in st.session_state:
        st.session_state[name_key] = default_value
    elif previous_default != default_value and current_value in {"", previous_default}:
        st.session_state[name_key] = default_value
    st.session_state[marker_key] = default_value


def _sync_select_default(key: str, options: list[str], default_value: str) -> None:
    if not options:
        st.session_state.pop(key, None)
        return
    current_value = str(st.session_state.get(key) or "")
    if current_value not in options:
        st.session_state[key] = default_value if default_value in options else options[0]


def _preferred_symbol_name(inspection: dict[str, object]) -> str:
    scenario_model_classes = list(inspection.get("scenario_model_classes") or [])
    if scenario_model_classes:
        return str(scenario_model_classes[0])
    symbols = list(inspection.get("symbols") or [])
    if not symbols:
        return ""
    return str((symbols[0] or {}).get("name") or "")


def _queue_selected_model(name: str | None) -> None:
    if name is None:
        st.session_state.pop(MODEL_CATALOG_PENDING_SELECTED_KEY, None)
        return
    st.session_state[MODEL_CATALOG_PENDING_SELECTED_KEY] = str(name)


def _apply_pending_selection(catalog_names: list[str]) -> None:
    pending = st.session_state.pop(MODEL_CATALOG_PENDING_SELECTED_KEY, None)
    if pending is not None and pending in catalog_names:
        st.session_state[MODEL_CATALOG_SELECTED_KEY] = pending
    selected = st.session_state.get(MODEL_CATALOG_SELECTED_KEY)
    if selected not in catalog_names and catalog_names:
        st.session_state[MODEL_CATALOG_SELECTED_KEY] = catalog_names[0]


def _catalog_frame() -> pd.DataFrame:
    rows = list_model_catalog()
    frame = pd.DataFrame(rows)
    return frame.drop(columns=["status", "removable"], errors="ignore")


def _selectable_catalog_names(catalog_frame: pd.DataFrame) -> list[str]:
    if catalog_frame.empty:
        return []
    return catalog_frame["name"].astype(str).tolist()


def _filtered_discovered_plugins(catalog_frame: pd.DataFrame) -> pd.DataFrame:
    plugin_frame = discover_plugins_df()
    plugin_error = plugin_frame.attrs.get("error")
    existing_refs = {
        ("plugin", row["reference_value"])
        for row in catalog_frame.to_dict(orient="records")
        if str(row.get("reference_kind") or "") == "plugin"
    }
    if plugin_frame.empty:
        return plugin_frame
    filtered = plugin_frame.loc[plugin_frame["source"] != "builtin"].copy()
    if filtered.empty:
        return filtered
    filtered = filtered.loc[
        ~filtered["plugin"].astype(str).map(lambda value: ("plugin", value) in existing_refs)
    ].copy()
    filtered = filtered.rename(
        columns={
            "plugin": "reference",
            "display_name": "display name",
        }
    )
    filtered.attrs["error"] = plugin_error
    return filtered


def _render_selected_model_detail(selected_name: str) -> None:
    detail = describe_catalog_model(selected_name)
    entry = detail["entry"]
    resolution = detail["resolution"]
    manifest = detail["manifest"]
    parameters = detail["parameters"]

    st.subheader("Selected model")
    title = manifest.get("display_name") or entry["name"]
    st.markdown(f"**{title}**")
    if entry.get("description"):
        st.caption(entry["description"])
    elif manifest.get("description"):
        st.caption(str(manifest["description"]))

    remove_disabled = not detail["removable"]
    if st.button(
        "Remove from catalog",
        disabled=remove_disabled,
        use_container_width=True,
        help="Built-in models are always available and cannot be removed." if remove_disabled else None,
    ):
        try:
            delete_catalog_model(entry["name"])
        except Exception as exc:
            _set_flash("error", f"Could not remove model: {exc}")
        else:
            _set_flash("success", f"Removed '{entry['name']}' from the catalog.")
            _queue_selected_model(None)
            st.session_state.pop(MODEL_CATALOG_SELECTED_KEY, None)
        st.rerun()

    if resolution.get("error"):
        st.error(f"Resolution failed: {resolution['error']}")

    st.subheader("Parameters")
    explicit_rows, accepts_varargs = _simplified_parameter_rows(parameters)
    if explicit_rows:
        st.dataframe(pd.DataFrame(explicit_rows), use_container_width=True, hide_index=True)
        if accepts_varargs:
            st.caption("This model also accepts additional free-form params.")
    elif accepts_varargs:
        st.info("This model accepts additional free-form params.")
    else:
        st.info("No constructor or factory parameters were discovered for this model.")


def _render_plugin_add_form(catalog_frame: pd.DataFrame) -> None:
    st.subheader("Add discovered plugin")
    plugin_frame = _filtered_discovered_plugins(catalog_frame)
    plugin_error = plugin_frame.attrs.get("error")
    if plugin_error:
        st.error(f"Plugin discovery failed: {plugin_error}")
    if plugin_frame.empty:
        st.info("No discoverable plugins are available to add right now.")
        return

    st.dataframe(plugin_frame, use_container_width=True, hide_index=True)
    references = plugin_frame["reference"].astype(str).tolist()
    selected_reference = st.selectbox("Discovered plugin", options=references, key="model_catalog.add.plugin_reference")
    default_name = selected_reference
    _sync_name_default(MODEL_CATALOG_PLUGIN_NAME_KEY, MODEL_CATALOG_PLUGIN_NAME_DEFAULT_KEY, default_name)
    add_cols = st.columns([2, 2, 1])
    catalog_name = add_cols[0].text_input("Catalog name", key=MODEL_CATALOG_PLUGIN_NAME_KEY)
    description = add_cols[1].text_input("Description", value="", key="model_catalog.add.plugin_description")
    if add_cols[2].button("Add plugin", use_container_width=True):
        entry = {
            "name": catalog_name,
            "description": description,
            "reference": {
                "kind": "plugin",
                "value": selected_reference,
            },
            "params": {},
            "metadata": {},
        }
        detail = describe_catalog_model_entry(entry)
        error = detail["resolution"].get("error")
        if error:
            _set_flash("error", f"Could not add plugin: {error}")
        else:
            try:
                save_catalog_model(entry)
            except Exception as exc:
                _set_flash("error", f"Could not add plugin: {exc}")
            else:
                _queue_selected_model(catalog_name)
                _set_flash("success", f"Added '{catalog_name}' to the model catalog.")
        st.rerun()


def _render_entrypoint_add_form() -> None:
    st.subheader("Add entrypoint")
    st.caption("Point to a Python file from your development repo. The catalog will infer the main public builder automatically.")
    entrypoint = ""
    file_path = ""
    repo_root = st.text_input(
        "Model repository",
        value=st.session_state.get(MODEL_CATALOG_ENTRYPOINT_REPO_ROOT_KEY, str(Path.cwd())),
        placeholder="/path/to/your/model/repo",
        key=MODEL_CATALOG_ENTRYPOINT_REPO_ROOT_KEY,
    )
    if repo_root:
        try:
            candidates = find_repo_scenario_model_candidates(repo_root)
        except Exception as exc:
            st.error(f"Could not browse repository: {exc}")
        else:
            if candidates:
                file_options = list(dict.fromkeys(candidate["path"] for candidate in candidates))
                _sync_select_default(
                    MODEL_CATALOG_ENTRYPOINT_FILE_KEY,
                    file_options,
                    str(candidates[0]["path"]),
                )
                file_path = st.selectbox(
                    "Detected model file",
                    options=file_options,
                    format_func=lambda value: _display_path(value),
                    key=MODEL_CATALOG_ENTRYPOINT_FILE_KEY,
                )
                st.caption("The first file exposing a ScenarioModel subclass is selected by default.")
            else:
                discovered_files = [str(path) for path in list_entrypoint_python_files(search_root=repo_root)]
                if discovered_files:
                    _sync_select_default(
                        MODEL_CATALOG_ENTRYPOINT_FILE_KEY,
                        discovered_files,
                        str(discovered_files[0]),
                    )
                    file_path = st.selectbox(
                        "Repository file",
                        options=discovered_files,
                        format_func=lambda value: _display_path(value),
                        key=MODEL_CATALOG_ENTRYPOINT_FILE_KEY,
                    )
                else:
                    st.info("No Python files were found under this repository root.")
    if file_path:
        try:
            inspection = inspect_entrypoint_python_file(file_path)
        except Exception as exc:
            st.error(f"Could not inspect file: {exc}")
        else:
            symbols = list(inspection.get("symbols") or [])
            if not symbols:
                st.warning("No public functions or classes were found in this file.")
            else:
                inferred_symbol = _preferred_symbol_name(inspection) or str(symbols[0]["name"])
                entrypoint = build_file_entrypoint_value(inspection["path"], inferred_symbol)
                st.caption(f"Using inferred symbol `{inferred_symbol}`")
                st.caption(f"Derived entrypoint: {entrypoint}")
    default_name = _entrypoint_default_name(entrypoint)
    _sync_name_default(MODEL_CATALOG_ENTRYPOINT_NAME_KEY, MODEL_CATALOG_ENTRYPOINT_NAME_DEFAULT_KEY, default_name)
    add_cols = st.columns([2, 2, 1])
    catalog_name = add_cols[0].text_input("Catalog name", key=MODEL_CATALOG_ENTRYPOINT_NAME_KEY)
    description = add_cols[1].text_input("Description", value="", key="model_catalog.add.entrypoint_description")
    if add_cols[2].button("Add entrypoint", use_container_width=True):
        entry = {
            "name": catalog_name,
            "description": description,
            "reference": {
                "kind": "entrypoint",
                "value": entrypoint,
            },
            "params": {},
            "metadata": {},
        }
        detail = describe_catalog_model_entry(entry)
        error = detail["resolution"].get("error")
        if error:
            _set_flash("error", f"Could not add entrypoint: {error}")
        else:
            try:
                save_catalog_model(entry)
            except Exception as exc:
                _set_flash("error", f"Could not add entrypoint: {exc}")
            else:
                _queue_selected_model(catalog_name)
                _set_flash("success", f"Added '{catalog_name}' to the model catalog.")
        st.rerun()


def render() -> None:
    st.header("Model Catalog")
    st.caption("Browse the current model catalog, add discoverable plugins or manual entrypoints, and inspect each model's parameters.")
    _render_flash()

    catalog_frame = _catalog_frame()
    if catalog_frame.empty:
        st.warning("The model catalog is empty.")
        return

    left_col, right_col = st.columns([1.4, 1.0], gap="large")
    with left_col:
        st.subheader("Current catalog")
        st.dataframe(catalog_frame, use_container_width=True, hide_index=True)

        catalog_names = _selectable_catalog_names(catalog_frame)
        _apply_pending_selection(catalog_names)
        st.selectbox(
            "Catalog entry",
            options=catalog_names,
            key=MODEL_CATALOG_SELECTED_KEY,
        )

        st.divider()
        add_tabs = st.tabs(["Discover plugin", "Entrypoint"])
        with add_tabs[0]:
            _render_plugin_add_form(catalog_frame)
        with add_tabs[1]:
            _render_entrypoint_add_form()

    with right_col:
        _render_selected_model_detail(str(st.session_state[MODEL_CATALOG_SELECTED_KEY]))
