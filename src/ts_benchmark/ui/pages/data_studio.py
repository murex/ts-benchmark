"""Dataset authoring page for the Streamlit UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ts_benchmark.dataset.providers.synthetic import RegimeSwitchingFactorSVGenerator

from ..services.datasets import (
    SUPPORTED_DATASET_SOURCES,
    delete_saved_dataset,
    default_dataset_dict,
    find_benchmark_configs_using_dataset,
    inspect_tabular_source,
    list_saved_datasets,
    load_saved_dataset,
    normalize_dataset_dict,
    save_dataset_definition,
    store_uploaded_dataset_file,
    switch_dataset_source,
)
from ..state import get_current_config, set_current_config

SOURCE_LABELS = {
    "synthetic": "Synthetic generator",
    "csv": "CSV file",
    "parquet": "Parquet file",
}
TABULAR_LAYOUT_LABELS = {
    "wide": "Wide",
    "long": "Long",
}
DATASET_SAVE_MESSAGE_KEY = "data_studio.saved_dataset_message"
DATA_STUDIO_SECTION_KEY = "data_studio.section"
DATA_STUDIO_DATASET_VIEW_KEY = "data_studio.dataset_view"
DATA_STUDIO_PENDING_SECTION_KEY = "data_studio.pending_section"
DATA_STUDIO_PENDING_DATASET_VIEW_KEY = "data_studio.pending_dataset_view"

TABULAR_VALUE_MODE_LABELS = {
    "price": "Price",
    "log_price": "Log price",
    "return": "Return",
    "log_return": "Log return",
}


def _render_section_heading(title: str, *, caption: str | None = None, level: str = "section") -> None:
    font_size = "1.05rem" if level == "section" else "0.96rem"
    margin_bottom = "0.35rem" if level == "section" else "0.25rem"
    st.markdown(
        (
            f"<div style='font-size:{font_size}; font-weight:600; margin:0.2rem 0 {margin_bottom} 0;'>"
            f"{title}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def _dataset_from_config() -> dict[str, Any]:
    config = get_current_config()
    benchmark = dict(config.get("benchmark") or {})
    dataset = benchmark.get("dataset")
    if not dataset:
        dataset = default_dataset_dict()
    return normalize_dataset_dict(dataset)


def _sync_dataset_to_config(dataset: dict[str, Any]) -> None:
    config = get_current_config()
    benchmark = dict(config.get("benchmark") or {})
    benchmark["dataset"] = normalize_dataset_dict(dataset)
    config["benchmark"] = benchmark
    set_current_config(config)


def _parse_optional_int(value: Any) -> tuple[int | None, str | None]:
    text = str(value or "").strip()
    if not text:
        return None, None
    try:
        return int(text), None
    except ValueError:
        return None, "Seed must be an integer or blank."


def _resize_float_vector(raw: Any, defaults: list[float]) -> list[float]:
    values = list(raw) if isinstance(raw, list) else []
    out: list[float] = []
    for index, default in enumerate(defaults):
        if index < len(values):
            try:
                out.append(float(values[index]))
            except (TypeError, ValueError):
                out.append(float(default))
        else:
            out.append(float(default))
    return out


def _resize_float_matrix(raw: Any, defaults: list[list[float]]) -> list[list[float]]:
    rows = raw if isinstance(raw, list) else []
    out: list[list[float]] = []
    for row_index, default_row in enumerate(defaults):
        candidate_row = rows[row_index] if row_index < len(rows) and isinstance(rows[row_index], list) else []
        out.append(_resize_float_vector(candidate_row, list(default_row)))
    return out


def _vector_editor(
    *,
    label: str,
    values: list[float],
    row_labels: list[str],
    key: str,
) -> list[float]:
    st.caption(label)
    frame = pd.DataFrame({"field": row_labels, "value": values})
    edited = st.data_editor(
        frame,
        key=key,
        disabled=["field"],
        hide_index=True,
        use_container_width=True,
    )
    series = pd.to_numeric(edited["value"], errors="coerce")
    fallback = pd.Series(values, index=series.index, dtype=float)
    return series.fillna(fallback).astype(float).tolist()


def _matrix_editor(
    *,
    label: str,
    values: list[list[float]],
    row_labels: list[str],
    column_labels: list[str],
    key: str,
) -> list[list[float]]:
    st.caption(label)
    frame = pd.DataFrame(values, index=row_labels, columns=column_labels, dtype=float)
    edited = st.data_editor(
        frame,
        key=key,
        use_container_width=True,
    )
    numeric = edited.apply(pd.to_numeric, errors="coerce")
    fallback = pd.DataFrame(values, index=row_labels, columns=column_labels, dtype=float)
    return numeric.fillna(fallback).astype(float).values.tolist()


def _coerce_option(value: Any, options: list[str], default: str) -> str:
    text = str(value or "").strip()
    if text in options:
        return text
    return default


def _format_frequency(value: Any) -> str:
    text = str(value or "").strip()
    labels = {
        "B": "Business day",
        "D": "Calendar day",
        "W": "Weekly",
        "M": "Month end",
        "MS": "Month start",
        "Q": "Quarter end",
        "QS": "Quarter start",
        "Y": "Year end",
        "YS": "Year start",
    }
    if not text:
        return "Not set"
    return f"{labels[text]} ({text})" if text in labels else text


def _default_data_studio_sections() -> None:
    st.session_state.setdefault(DATA_STUDIO_SECTION_KEY, "Catalog")
    st.session_state.setdefault(DATA_STUDIO_DATASET_VIEW_KEY, "Definition")


def _uploaded_file_signature(uploaded: Any) -> str:
    return ":".join(
        [
            str(getattr(uploaded, "name", "")),
            str(getattr(uploaded, "size", "")),
            str(getattr(uploaded, "file_id", "")),
        ]
    )


def _handle_uploaded_tabular_file(source: str, uploaded: Any) -> str | None:
    if uploaded is None:
        return None

    marker_key = f"data_studio.{source}.upload_marker"
    path_key = f"data_studio.{source}.path"
    saved_path_key = f"data_studio.{source}.uploaded_path"
    signature = _uploaded_file_signature(uploaded)

    if st.session_state.get(marker_key) != signature:
        stored_path = store_uploaded_dataset_file(
            filename=uploaded.name,
            content=uploaded.getvalue(),
        )
        st.session_state[path_key] = str(stored_path)
        st.session_state[saved_path_key] = str(stored_path)
        st.session_state[marker_key] = signature

    return st.session_state.get(saved_path_key)


def _apply_pending_data_studio_navigation() -> None:
    pending_section = st.session_state.pop(DATA_STUDIO_PENDING_SECTION_KEY, None)
    pending_view = st.session_state.pop(DATA_STUDIO_PENDING_DATASET_VIEW_KEY, None)
    if pending_section is not None:
        st.session_state[DATA_STUDIO_SECTION_KEY] = str(pending_section)
    if pending_view is not None:
        st.session_state[DATA_STUDIO_DATASET_VIEW_KEY] = str(pending_view)


def _reset_dataset_editor_widget_state() -> None:
    persistent_keys = {
        DATASET_SAVE_MESSAGE_KEY,
        DATA_STUDIO_SECTION_KEY,
        DATA_STUDIO_DATASET_VIEW_KEY,
        DATA_STUDIO_PENDING_SECTION_KEY,
        DATA_STUDIO_PENDING_DATASET_VIEW_KEY,
    }
    for key in list(st.session_state):
        if key.startswith("data_studio.") and key not in persistent_keys:
            del st.session_state[key]


def _open_dataset_definition(dataset: dict[str, Any]) -> None:
    _reset_dataset_editor_widget_state()
    _sync_dataset_to_config(dataset)
    st.session_state[DATA_STUDIO_PENDING_SECTION_KEY] = "Dataset"
    st.session_state[DATA_STUDIO_PENDING_DATASET_VIEW_KEY] = "Definition"


def _new_dataset() -> None:
    _open_dataset_definition(default_dataset_dict())


def _tabular_value_mode_from_config(
    config: dict[str, Any],
    semantics: dict[str, Any],
) -> str:
    explicit = str(config.get("value_type") or semantics.get("target_kind") or "").strip().lower()
    if explicit in TABULAR_VALUE_MODE_LABELS:
        return explicit

    target_kind = str(config.get("value_type") or semantics.get("target_kind") or "returns").strip().lower()
    return_kind = str(config.get("return_kind") or semantics.get("return_kind") or "simple").strip().lower()

    if target_kind in {"prices", "price"}:
        return "price" if return_kind == "simple" else "log_price"
    if target_kind in {"log_prices", "log_price"}:
        return "log_price"
    if target_kind in {"returns", "return"}:
        return "return" if return_kind == "simple" else "log_return"
    if target_kind in {"log_returns", "log_return"}:
        return "log_return"
    return "return"


def _apply_tabular_value_mode(
    *,
    config: dict[str, Any],
    semantics: dict[str, Any],
    mode: str,
) -> None:
    normalized = str(mode).strip().lower()
    if normalized == "price":
        config["value_type"] = "price"
        config["return_kind"] = "simple"
        semantics["target_kind"] = "price"
        semantics["return_kind"] = "simple"
        return
    if normalized == "log_price":
        config["value_type"] = "log_price"
        config["return_kind"] = "log"
        semantics["target_kind"] = "log_price"
        semantics["return_kind"] = "log"
        return
    if normalized == "log_return":
        config["value_type"] = "log_return"
        config["return_kind"] = "log"
        semantics["target_kind"] = "log_return"
        semantics["return_kind"] = "log"
        return

    config["value_type"] = "return"
    config["return_kind"] = "simple"
    semantics["target_kind"] = "return"
    semantics["return_kind"] = "simple"


def _synthetic_default_params(n_assets: int) -> dict[str, Any]:
    generator = RegimeSwitchingFactorSVGenerator(n_assets=n_assets)
    return {
        "n_assets": int(generator.n_assets),
        "factor_loadings": generator.factor_loadings.astype(float).tolist(),
        "idio_scales": generator.idio_scales.astype(float).tolist(),
        "regime_drifts": generator.regime_drifts.astype(float).tolist(),
        "transition_matrix": generator.transition_matrix.astype(float).tolist(),
        "market_log_var_means": generator.market_log_var_means.astype(float).tolist(),
        "idio_log_var_means": generator.idio_log_var_means.astype(float).tolist(),
        "market_phi": float(generator.market_phi),
        "idio_phi": float(generator.idio_phi),
        "market_vol_of_vol": float(generator.market_vol_of_vol),
        "idio_vol_of_vol": float(generator.idio_vol_of_vol),
        "market_leverage": float(generator.market_leverage),
        "idio_leverage": float(generator.idio_leverage),
        "student_df": float(generator.student_df),
        "seed": generator.seed,
    }


def _render_synthetic_dataset_editor(dataset: dict[str, Any]) -> None:
    provider = dataset.setdefault("provider", {})
    config = dict(provider.get("config") or {})
    params = dict(config.get("params") or {})

    st.subheader("Synthetic generator")
    st.selectbox(
        "Values represent",
        options=["return"],
        index=0,
        format_func=lambda value: TABULAR_VALUE_MODE_LABELS.get(value, str(value).title()),
        disabled=True,
        key="data_studio.synthetic.value_mode",
    )
    generator_name = st.selectbox(
        "Generator",
        options=["regime_switching_factor_sv"],
        format_func=lambda value: "Regime-switching factor stochastic volatility",
        index=0,
        key="data_studio.synthetic.generator",
    )
    n_assets = int(
        st.number_input(
            "Number of assets",
            min_value=1,
            max_value=128,
            value=int(params.get("n_assets") or 6),
            step=1,
            key="data_studio.synthetic.n_assets",
        )
    )
    defaults = _synthetic_default_params(n_assets)

    seed_value, seed_error = _parse_optional_int(
        st.text_input(
            "Generator seed",
            value="" if params.get("seed") is None else str(params.get("seed")),
            help="Leave blank for nondeterministic samples.",
            key="data_studio.synthetic.seed",
        )
    )
    if seed_error:
        st.error(seed_error)

    basic_cols = st.columns(3)
    params["student_df"] = float(
        basic_cols[0].number_input(
            "Student-t dof",
            min_value=2.01,
            value=float(params.get("student_df", defaults["student_df"])),
            key="data_studio.synthetic.student_df",
        )
    )
    params["market_phi"] = float(
        basic_cols[1].number_input(
            "Market phi",
            min_value=0.0,
            max_value=0.9999,
            value=float(params.get("market_phi", defaults["market_phi"])),
            key="data_studio.synthetic.market_phi",
        )
    )
    params["idio_phi"] = float(
        basic_cols[2].number_input(
            "Idiosyncratic phi",
            min_value=0.0,
            max_value=0.9999,
            value=float(params.get("idio_phi", defaults["idio_phi"])),
            key="data_studio.synthetic.idio_phi",
        )
    )

    vol_cols = st.columns(4)
    params["market_vol_of_vol"] = float(
        vol_cols[0].number_input(
            "Market vol-of-vol",
            min_value=0.0,
            value=float(params.get("market_vol_of_vol", defaults["market_vol_of_vol"])),
            key="data_studio.synthetic.market_vol_of_vol",
        )
    )
    params["idio_vol_of_vol"] = float(
        vol_cols[1].number_input(
            "Idio vol-of-vol",
            min_value=0.0,
            value=float(params.get("idio_vol_of_vol", defaults["idio_vol_of_vol"])),
            key="data_studio.synthetic.idio_vol_of_vol",
        )
    )
    params["market_leverage"] = float(
        vol_cols[2].number_input(
            "Market leverage",
            value=float(params.get("market_leverage", defaults["market_leverage"])),
            key="data_studio.synthetic.market_leverage",
        )
    )
    params["idio_leverage"] = float(
        vol_cols[3].number_input(
            "Idio leverage",
            value=float(params.get("idio_leverage", defaults["idio_leverage"])),
            key="data_studio.synthetic.idio_leverage",
        )
    )

    asset_labels = [f"Asset {index + 1}" for index in range(n_assets)]
    regime_labels = ["Calm", "Stress"]
    factor_loadings = _resize_float_vector(params.get("factor_loadings"), defaults["factor_loadings"])
    idio_scales = _resize_float_vector(params.get("idio_scales"), defaults["idio_scales"])
    regime_drifts = _resize_float_vector(params.get("regime_drifts"), defaults["regime_drifts"])
    market_log_var_means = _resize_float_vector(
        params.get("market_log_var_means"),
        defaults["market_log_var_means"],
    )
    transition_matrix = _resize_float_matrix(
        params.get("transition_matrix"),
        defaults["transition_matrix"],
    )
    idio_log_var_means = _resize_float_matrix(
        params.get("idio_log_var_means"),
        defaults["idio_log_var_means"],
    )

    with st.expander("Regime parameters", expanded=False):
        params["regime_drifts"] = _vector_editor(
            label="Regime drifts",
            values=regime_drifts,
            row_labels=regime_labels,
            key="data_studio.synthetic.regime_drifts",
        )
        params["market_log_var_means"] = _vector_editor(
            label="Market log-variance means",
            values=market_log_var_means,
            row_labels=regime_labels,
            key="data_studio.synthetic.market_log_var_means",
        )
        params["transition_matrix"] = _matrix_editor(
            label="Transition matrix",
            values=transition_matrix,
            row_labels=regime_labels,
            column_labels=regime_labels,
            key="data_studio.synthetic.transition_matrix",
        )

    with st.expander("Asset parameters", expanded=False):
        params["factor_loadings"] = _vector_editor(
            label="Factor loadings",
            values=factor_loadings,
            row_labels=asset_labels,
            key="data_studio.synthetic.factor_loadings",
        )
        params["idio_scales"] = _vector_editor(
            label="Idiosyncratic scales",
            values=idio_scales,
            row_labels=asset_labels,
            key="data_studio.synthetic.idio_scales",
        )
        params["idio_log_var_means"] = _matrix_editor(
            label="Idiosyncratic log-variance means",
            values=idio_log_var_means,
            row_labels=regime_labels,
            column_labels=asset_labels,
            key="data_studio.synthetic.idio_log_var_means",
        )

    provider["kind"] = "synthetic"
    provider["config"] = {
        "generator": generator_name,
        "params": {
            "n_assets": n_assets,
            "factor_loadings": params["factor_loadings"],
            "idio_scales": params["idio_scales"],
            "regime_drifts": params["regime_drifts"],
            "transition_matrix": params["transition_matrix"],
            "market_log_var_means": params["market_log_var_means"],
            "idio_log_var_means": params["idio_log_var_means"],
            "market_phi": params["market_phi"],
            "idio_phi": params["idio_phi"],
            "market_vol_of_vol": params["market_vol_of_vol"],
            "idio_vol_of_vol": params["idio_vol_of_vol"],
            "market_leverage": params["market_leverage"],
            "idio_leverage": params["idio_leverage"],
            "student_df": params["student_df"],
            "seed": seed_value,
        },
    }
    dataset["schema"] = {
        "layout": "tensor",
        "frequency": st.text_input(
            "Frequency",
            value=str(dataset.get("schema", {}).get("frequency") or "B"),
            help="Pandas offset alias. Common values: B=business day, D=calendar day, W=weekly, M=month end.",
            key="data_studio.synthetic.frequency",
        ).strip()
        or "B",
    }
    dataset["semantics"] = dict(dataset.get("semantics") or {})
    dataset["metadata"] = dict(dataset.get("metadata") or {})


def _default_time_column(columns: list[str], current: Any) -> str | None:
    current_text = str(current or "").strip()
    if current_text in columns:
        return current_text
    for candidate in columns:
        lowered = candidate.lower()
        if lowered in {"date", "datetime", "timestamp", "time"}:
            return candidate
    return None


def _render_tabular_dataset_editor(dataset: dict[str, Any]) -> dict[str, Any] | None:
    source = str(dataset.get("provider", {}).get("kind") or "csv")
    provider = dataset.setdefault("provider", {})
    config = dict(provider.get("config") or {})
    schema = dict(dataset.get("schema") or {})
    semantics = dict(dataset.get("semantics") or {})
    path_key = f"data_studio.{source}.path"

    if path_key not in st.session_state:
        st.session_state[path_key] = str(config.get("path") or "")

    st.subheader(f"{SOURCE_LABELS[source]} settings")
    uploaded = st.file_uploader(
        f"Upload {source.upper()} file",
        type=[source],
        key=f"data_studio.{source}.upload",
    )
    saved_upload_path = _handle_uploaded_tabular_file(source, uploaded)
    if saved_upload_path:
        st.caption(f"Uploaded file saved to {saved_upload_path}")

    config["path"] = st.text_input(
        "Dataset file path",
        key=path_key,
    ).strip()

    settings_cols = st.columns(3)
    value_mode_options = list(TABULAR_VALUE_MODE_LABELS)
    value_mode = _tabular_value_mode_from_config(config, semantics)
    selected_value_mode = settings_cols[0].selectbox(
        "Values represent",
        options=value_mode_options,
        index=value_mode_options.index(value_mode),
        format_func=lambda value: TABULAR_VALUE_MODE_LABELS[value],
        key=f"data_studio.{source}.value_mode",
    )
    _apply_tabular_value_mode(config=config, semantics=semantics, mode=selected_value_mode)
    dropna = _coerce_option(config.get("dropna"), ["any", "none"], "any")
    config["dropna"] = settings_cols[1].selectbox(
        "Missing values",
        options=["any", "none"],
        index=["any", "none"].index(dropna),
        format_func=lambda value: "Drop rows with NA" if value == "any" else "Fail on NA",
        key=f"data_studio.{source}.dropna",
    )
    config["sort_ascending"] = settings_cols[2].checkbox(
        "Sort ascending",
        value=bool(config.get("sort_ascending", True)),
        key=f"data_studio.{source}.sort_ascending",
    )

    layout_options = list(TABULAR_LAYOUT_LABELS)
    current_layout = _coerce_option(schema.get("layout"), layout_options, "wide")
    schema["layout"] = st.selectbox(
        "Layout",
        options=layout_options,
        index=layout_options.index(current_layout),
        format_func=lambda value: TABULAR_LAYOUT_LABELS[value],
        key=f"data_studio.{source}.layout",
        help="Wide: one column per series/variate. Long: one row per timestamp and series identifier.",
    )

    schema["frequency"] = (
        st.text_input(
            "Frequency",
            value=str(schema.get("frequency") or "B"),
            help="Pandas offset alias. Common values: B=business day, D=calendar day, W=weekly, M=month end.",
            key=f"data_studio.{source}.frequency",
        ).strip()
        or "B"
    )

    provider["kind"] = source
    provider["config"] = config
    dataset["schema"] = schema
    dataset["semantics"] = semantics
    dataset["metadata"] = dict(dataset.get("metadata") or {})

    if not config["path"]:
        st.info("Upload a file or enter a file path to inspect columns and map the dataset schema.")
        return None

    try:
        inspection = inspect_tabular_source(path=config["path"], source=source)
    except Exception as exc:
        st.error(f"Could not inspect the dataset file: {type(exc).__name__}: {exc}")
        return None

    columns = list(inspection["columns"])
    numeric_columns = list(inspection["numeric_columns"])
    column_frame = pd.DataFrame(
        {
            "column": columns,
            "numeric": [column in numeric_columns for column in columns],
        }
    )
    st.caption(f"Detected {len(columns)} columns from {inspection['path']}")
    st.dataframe(column_frame, hide_index=True, use_container_width=True)
    st.caption("File preview")
    st.dataframe(inspection["preview"], hide_index=True, use_container_width=True)

    time_default = _default_time_column(columns, schema.get("time_column"))
    time_options = ["<none>"] + columns
    time_choice = st.selectbox(
        "Time column",
        options=time_options,
        index=time_options.index(time_default) if time_default in time_options else 0,
        key=f"data_studio.{source}.time_column",
    )
    schema["time_column"] = None if time_choice == "<none>" else time_choice

    if schema["layout"] == "long":
        series_options = [column for column in columns if column != schema["time_column"]]
        series_defaults = [
            column for column in list(schema.get("series_id_columns") or []) if column in series_options
        ]
        schema["series_id_columns"] = st.multiselect(
            "Series ID columns",
            options=series_options,
            default=series_defaults,
            key=f"data_studio.{source}.series_id_columns",
            help="These columns identify the series or variate on each row.",
        )

        value_options = [
            column
            for column in numeric_columns
            if column != schema["time_column"] and column not in schema["series_id_columns"]
        ]
        current_value_column = str(config.get("value_column") or "").strip()
        if current_value_column not in value_options:
            current_value_column = value_options[0] if value_options else ""

        if value_options:
            config["value_column"] = st.selectbox(
                "Value column",
                options=value_options,
                index=value_options.index(current_value_column),
                key=f"data_studio.{source}.value_column",
                help="Numeric column to pivot into the multivariate panel.",
            )
        else:
            config["value_column"] = ""
            st.warning("Select at least one series ID column and keep one numeric value column for the series values.")

        schema["target_columns"] = []
        schema["feature_columns"] = []
        schema["static_feature_columns"] = []
        st.info("Target columns are derived from the series ID values after pivoting the long table to wide format.")
    else:
        config.pop("value_column", None)
        schema["series_id_columns"] = []

        target_options = [
            column
            for column in numeric_columns
            if column != schema["time_column"]
        ]
        target_defaults = [
            column for column in list(schema.get("target_columns") or []) if column in target_options
        ] or target_options
        schema["target_columns"] = st.multiselect(
            "Target columns",
            options=target_options,
            default=target_defaults,
            key=f"data_studio.{source}.target_columns",
        )
        if not target_options:
            st.warning("No numeric target columns were detected yet. Choose a different file or adjust the schema mapping.")

        feature_options = [
            column
            for column in columns
            if column != schema["time_column"]
            and column not in schema["target_columns"]
        ]
        feature_defaults = [column for column in list(schema.get("feature_columns") or []) if column in feature_options]
        schema["feature_columns"] = st.multiselect(
            "Feature columns",
            options=feature_options,
            default=feature_defaults,
            key=f"data_studio.{source}.feature_columns",
        )

        static_options = [column for column in feature_options if column not in schema["feature_columns"]]
        static_defaults = [
            column for column in list(schema.get("static_feature_columns") or []) if column in static_options
        ]
        schema["static_feature_columns"] = st.multiselect(
            "Static feature columns",
            options=static_options,
            default=static_defaults,
            key=f"data_studio.{source}.static_feature_columns",
        )

    dataset["schema"] = schema
    dataset["semantics"] = semantics
    return inspection


def _render_catalog_tab() -> None:
    header_cols = st.columns([4, 1])
    header_cols[0].subheader("Catalog")
    if header_cols[1].button(
        "New",
        icon=":material/add:",
        type="primary",
        width="stretch",
        key="data_studio.catalog.new",
    ):
        _new_dataset()
        st.rerun()

    rows = list_saved_datasets()
    if not rows:
        st.info("No saved datasets yet. Build one in the Dataset tab and save it to the catalog.")
        return

    table_header_cols = st.columns([2.0, 3.0, 1.5, 1.8, 0.8, 0.8])
    table_header_cols[0].caption("Name")
    table_header_cols[1].caption("Description")
    table_header_cols[2].caption("Source")
    table_header_cols[3].caption("Frequency")
    table_header_cols[4].caption("Open")
    table_header_cols[5].caption("Delete")

    for row in rows:
        dataset_name = str(row.get("name") or "")
        selected_dataset = load_saved_dataset(dataset_name)
        usage_paths = find_benchmark_configs_using_dataset(selected_dataset["name"])

        row_cols = st.columns([2.0, 3.0, 1.5, 1.8, 0.8, 0.8])
        row_cols[0].write(f"**{dataset_name}**")
        row_cols[1].write(str(row.get("description") or ""))
        row_cols[2].write(SOURCE_LABELS.get(str(row.get("source") or ""), str(row.get("source") or "")))
        row_cols[3].write(_format_frequency(row.get("frequency")))

        if row_cols[4].button(
            " ",
            key=f"data_studio.catalog.open.{dataset_name}",
            icon=":material/open_in_new:",
            type="tertiary",
            use_container_width=True,
            help="Open this dataset in Dataset > Definition.",
        ):
            _open_dataset_definition(selected_dataset)
            st.rerun()

        if row_cols[5].button(
            " ",
            key=f"data_studio.catalog.delete.{dataset_name}",
            icon=":material/delete:",
            type="tertiary",
            use_container_width=True,
            disabled=bool(usage_paths),
            help=None if not usage_paths else "Dataset is used by one or more saved benchmark configs.",
        ):
            deleted, blocked_by = delete_saved_dataset(dataset_name)
            if deleted:
                st.rerun()
            st.error("Dataset is still in use by: " + ", ".join(path.name for path in blocked_by))

        if usage_paths:
            st.caption("Used by benchmark configs: " + ", ".join(path.name for path in usage_paths))
        st.divider()


def _save_dataset(dataset: dict[str, Any]) -> None:
    _sync_dataset_to_config(dataset)
    save_dataset_definition(dataset)
    st.session_state[DATASET_SAVE_MESSAGE_KEY] = f"Saved dataset '{dataset['name']}' to the catalog."


def _configured_tabular_preview(dataset: dict[str, Any], preview: pd.DataFrame) -> pd.DataFrame:
    schema = dict(dataset.get("schema") or {})
    provider_config = dict(dataset.get("provider", {}).get("config") or {})
    layout = str(schema.get("layout") or "wide").strip().lower()
    time_column = str(schema.get("time_column") or "").strip()

    if layout == "long":
        series_id_columns = list(schema.get("series_id_columns") or [])
        value_column = str(provider_config.get("value_column") or "").strip()
        required = [column for column in [time_column, *series_id_columns, value_column] if column]
        if required:
            preview = preview.loc[:, [column for column in required if column in preview.columns]].copy()
        if time_column and series_id_columns and value_column and all(column in preview.columns for column in required):
            preview[time_column] = pd.to_datetime(preview[time_column], errors="coerce")
            preview = preview.dropna(subset=[time_column])
            if len(series_id_columns) == 1:
                preview["_series_key"] = preview[series_id_columns[0]].astype(str)
            else:
                preview["_series_key"] = preview.loc[:, series_id_columns].astype(str).agg("::".join, axis=1)
            preview = (
                preview.pivot(index=time_column, columns="_series_key", values=value_column)
                .sort_index()
                .reset_index()
            )
            preview.columns = [str(column) for column in preview.columns]
        return preview

    selected_columns: list[str] = []
    if time_column and time_column in preview.columns:
        selected_columns.append(time_column)
    selected_columns.extend(
        column
        for column in list(schema.get("target_columns") or [])
        if column in preview.columns and column not in selected_columns
    )
    selected_columns.extend(
        column
        for column in list(schema.get("feature_columns") or [])
        if column in preview.columns and column not in selected_columns
    )
    selected_columns.extend(
        column
        for column in list(schema.get("static_feature_columns") or [])
        if column in preview.columns and column not in selected_columns
    )
    if selected_columns:
        return preview.loc[:, selected_columns].copy()
    return preview


def _configured_tabular_preview_from_source(dataset: dict[str, Any], max_rows: int = 5000) -> pd.DataFrame:
    provider = dict(dataset.get("provider") or {})
    source = str(provider.get("kind") or "").strip().lower()
    if source not in {"csv", "parquet"}:
        raise ValueError(f"Unsupported tabular source: {source}")

    provider_config = dict(provider.get("config") or {})
    schema = dict(dataset.get("schema") or {})
    path = str(provider_config.get("path") or "").strip()
    if not path:
        raise FileNotFoundError("Dataset file path is not configured.")

    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    layout = str(schema.get("layout") or "wide").strip().lower()
    time_column = str(schema.get("time_column") or "").strip()
    read_kwargs = dict(provider_config.get("read_kwargs") or {})

    if layout == "long":
        series_id_columns = list(schema.get("series_id_columns") or [])
        value_column = str(provider_config.get("value_column") or "").strip()
        use_columns = [
            column
            for column in [time_column, *series_id_columns, value_column]
            if column
        ]
        if not use_columns:
            raise ValueError("Long-format preview requires time, series ID, and value columns.")
        if source == "csv":
            frame = pd.read_csv(dataset_path, usecols=use_columns, **read_kwargs)
        else:
            frame = pd.read_parquet(dataset_path, columns=use_columns, **read_kwargs)
        preview = _configured_tabular_preview(dataset, frame)
        if time_column and time_column in preview.columns:
            preview = preview.sort_values(time_column).reset_index(drop=True)
        return preview.head(max_rows)

    selected_columns: list[str] = []
    if time_column:
        selected_columns.append(time_column)
    selected_columns.extend(
        column
        for key in ("target_columns", "feature_columns", "static_feature_columns")
        for column in list(schema.get(key) or [])
        if column not in selected_columns
    )
    use_columns = selected_columns or None
    if source == "csv":
        frame = pd.read_csv(dataset_path, usecols=use_columns, nrows=max_rows, **read_kwargs)
    else:
        frame = pd.read_parquet(dataset_path, columns=use_columns, **read_kwargs).head(max_rows)
    return _configured_tabular_preview(dataset, frame)


def _build_preview_payload(dataset: dict[str, Any], tabular_inspection: dict[str, Any] | None) -> tuple[pd.DataFrame | None, str | None]:
    source = dataset.get("provider", {}).get("kind")
    if source == "synthetic":
        provider_config = dict(dataset.get("provider", {}).get("config") or {})
        params = dict(provider_config.get("params") or {})
        try:
            generator = RegimeSwitchingFactorSVGenerator(**params)
            simulation = generator.simulate(n_steps=64, seed=params.get("seed"))
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

        frame = pd.DataFrame(simulation.returns, columns=simulation.asset_names)
        frame.insert(0, "regime", [state.regime for state in simulation.states])
        return frame, None

    if source in {"csv", "parquet"}:
        try:
            return _configured_tabular_preview_from_source(dataset, max_rows=5000), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    return None, f"Unsupported dataset source: {source}"


def _render_preview_tab(dataset: dict[str, Any], preview_frame: pd.DataFrame | None, preview_error: str | None) -> None:
    _render_section_heading("Preview")
    preview_summary = pd.DataFrame(
        [
            {
                "dataset": str(dataset.get("name") or "<unsaved dataset>"),
                "source": SOURCE_LABELS.get(
                    str(dataset.get("provider", {}).get("kind") or ""),
                    str(dataset.get("provider", {}).get("kind") or ""),
                ),
                "frequency": _format_frequency(dataset.get("schema", {}).get("frequency")),
            }
        ]
    )
    st.dataframe(preview_summary, hide_index=True, use_container_width=True)
    description = str(dataset.get("description") or "").strip()
    if description:
        st.caption(description)

    if preview_error:
        st.error(f"Preview unavailable: {preview_error}")
        return
    if preview_frame is None or preview_frame.empty:
        st.info("No preview rows are available for this dataset.")
        return

    numeric_columns = preview_frame.select_dtypes(include="number").columns.tolist()
    time_column = str(dataset.get("schema", {}).get("time_column") or "").strip()
    preferred_columns = list(dataset.get("schema", {}).get("target_columns") or [])
    plot_candidates = [column for column in preferred_columns if column in preview_frame.columns]
    if not plot_candidates:
        plot_candidates = [column for column in numeric_columns if column != time_column and column != "regime"]
    default_plot_columns = plot_candidates[: min(6, len(plot_candidates))]

    if plot_candidates:
        _render_section_heading("Plot columns", level="subsection")
        selected_plot_columns = st.multiselect(
            "Plot columns",
            options=plot_candidates,
            default=default_plot_columns,
            key="data_studio.preview.plot_columns",
        )
        if selected_plot_columns:
            plot_frame = preview_frame.loc[:, selected_plot_columns].copy()
            if time_column and time_column in preview_frame.columns:
                timestamps = pd.to_datetime(preview_frame[time_column], errors="coerce")
                if timestamps.notna().any():
                    valid_mask = timestamps.notna().to_numpy()
                    plot_frame = plot_frame.loc[valid_mask].copy()
                    plot_frame.index = pd.Index(timestamps.loc[valid_mask], name=time_column)
            st.line_chart(plot_frame, use_container_width=True)

    st.dataframe(preview_frame, use_container_width=True)


def _series_frame_for_statistics(dataset: dict[str, Any], preview_frame: pd.DataFrame) -> pd.DataFrame:
    numeric = preview_frame.select_dtypes(include="number")
    preferred_columns = [
        column
        for column in list(dataset.get("schema", {}).get("target_columns") or [])
        if column in numeric.columns
    ]
    if preferred_columns:
        return numeric.loc[:, preferred_columns].copy()

    excluded = {"regime"}
    remaining = [column for column in numeric.columns if column not in excluded]
    return numeric.loc[:, remaining].copy()


def _render_data_quality_statistics(dataset: dict[str, Any], preview_frame: pd.DataFrame) -> None:
    time_column = str(dataset.get("schema", {}).get("time_column") or "").strip()
    rows = int(len(preview_frame))
    columns = int(len(preview_frame.columns))
    missing_cells = int(preview_frame.isna().sum().sum())
    total_cells = max(rows * columns, 1)
    completeness = 1.0 - (missing_cells / total_cells)
    duplicate_rows = int(preview_frame.duplicated().sum())

    duplicate_timestamps = 0
    if time_column and time_column in preview_frame.columns:
        timestamps = pd.to_datetime(preview_frame[time_column], errors="coerce")
        duplicate_timestamps = int(timestamps.dropna().duplicated().sum())

    quality_summary = pd.DataFrame(
        [
            {
                "rows": rows,
                "columns": columns,
                "missing_cells": missing_cells,
                "completeness": f"{completeness:.1%}",
                "duplicate_rows": duplicate_rows,
            }
        ]
    )
    st.dataframe(quality_summary, hide_index=True, use_container_width=True)
    if time_column and time_column in preview_frame.columns:
        st.caption(f"Duplicate timestamps in `{time_column}`: {duplicate_timestamps}")

    column_profile = pd.DataFrame(
        {
            "column": list(preview_frame.columns),
            "dtype": [str(dtype) for dtype in preview_frame.dtypes],
            "missing": [int(preview_frame[column].isna().sum()) for column in preview_frame.columns],
            "missing_pct": [
                float(preview_frame[column].isna().mean()) if rows else 0.0
                for column in preview_frame.columns
            ],
            "unique": [int(preview_frame[column].nunique(dropna=True)) for column in preview_frame.columns],
        }
    )
    _render_section_heading("Column quality profile", level="subsection")
    st.dataframe(column_profile, hide_index=True, use_container_width=True)


def _render_time_series_statistics(dataset: dict[str, Any], preview_frame: pd.DataFrame) -> None:
    series_frame = _series_frame_for_statistics(dataset, preview_frame)
    if series_frame.empty:
        st.info("The current dataset preview does not expose numeric series columns to analyze.")
        return

    variance = series_frame.var(ddof=1)
    correlation = series_frame.corr()
    if correlation.shape[0] > 1:
        corr_values = correlation.where(~pd.DataFrame(
            [[i == j for j in range(correlation.shape[1])] for i in range(correlation.shape[0])],
            index=correlation.index,
            columns=correlation.columns,
        )).stack()
        avg_abs_corr = float(corr_values.abs().mean()) if not corr_values.empty else 0.0
    else:
        avg_abs_corr = 0.0

    series_summary = pd.DataFrame(
        [
            {
                "series": int(series_frame.shape[1]),
                "observations": int(series_frame.shape[0]),
                "avg_variance": f"{float(variance.mean()):.4g}",
                "avg_abs_corr": f"{avg_abs_corr:.3f}",
            }
        ]
    )
    st.dataframe(series_summary, hide_index=True, use_container_width=True)

    series_stats = pd.DataFrame(
        {
            "series": list(series_frame.columns),
            "mean": series_frame.mean().to_list(),
            "variance": variance.to_list(),
            "std_dev": series_frame.std(ddof=1).to_list(),
            "min": series_frame.min().to_list(),
            "max": series_frame.max().to_list(),
            "skew": series_frame.skew().to_list(),
            "kurtosis": series_frame.kurt().to_list(),
            "lag1_autocorr": [float(series_frame[column].autocorr(lag=1)) for column in series_frame.columns],
        }
    )
    _render_section_heading("Per-series statistics", level="subsection")
    st.dataframe(series_stats, hide_index=True, use_container_width=True)

    _render_section_heading("Variance by series", level="subsection")
    st.bar_chart(series_stats.set_index("series")[["variance"]], use_container_width=True)

    _render_section_heading("Correlation matrix", level="subsection")
    st.dataframe(correlation.round(3), use_container_width=True)

    if correlation.shape[0] > 1:
        corr_pairs = []
        columns = list(correlation.columns)
        for left_index, left_name in enumerate(columns):
            for right_name in columns[left_index + 1 :]:
                corr_pairs.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "correlation": float(correlation.loc[left_name, right_name]),
                    }
                )
        top_pairs = (
            pd.DataFrame(corr_pairs)
            .assign(abs_correlation=lambda frame: frame["correlation"].abs())
            .sort_values("abs_correlation", ascending=False)
            .head(10)
            .drop(columns=["abs_correlation"])
        )
        _render_section_heading("Top absolute correlation pairs", level="subsection")
        st.dataframe(top_pairs, hide_index=True, use_container_width=True)


def _render_statistics_tab(dataset: dict[str, Any], preview_frame: pd.DataFrame | None, preview_error: str | None) -> None:
    _render_section_heading("Statistics")
    if preview_error:
        st.error(f"Statistics unavailable: {preview_error}")
        return
    if preview_frame is None or preview_frame.empty:
        st.info("No preview data is available to summarize.")
        return

    statistics_view = st.segmented_control(
        "Statistics view",
        options=["Data quality", "Time series"],
        key="data_studio.statistics.view",
        label_visibility="collapsed",
        width="stretch",
    )
    statistics_view = str(statistics_view or "Data quality")

    if statistics_view == "Data quality":
        _render_data_quality_statistics(dataset, preview_frame)
        return

    _render_time_series_statistics(dataset, preview_frame)


def render() -> None:
    st.header("Data Studio")
    st.caption("Define reusable datasets, save them to the catalog, and map files into benchmark-ready dataset definitions.")
    _default_data_studio_sections()
    _apply_pending_data_studio_navigation()

    dataset = _dataset_from_config()
    tabular_inspection: dict[str, Any] | None = None

    section = st.segmented_control(
        "Section",
        options=["Catalog", "Dataset"],
        key=DATA_STUDIO_SECTION_KEY,
        label_visibility="collapsed",
        width="stretch",
    )
    section = str(section or st.session_state.get(DATA_STUDIO_SECTION_KEY, "Catalog"))

    if section == "Catalog":
        _render_catalog_tab()

    if section == "Dataset":
        dataset_view = st.segmented_control(
            "Dataset view",
            options=["Definition", "Preview", "Statistics"],
            key=DATA_STUDIO_DATASET_VIEW_KEY,
            label_visibility="collapsed",
            width="stretch",
        )
        dataset_view = str(dataset_view or st.session_state.get(DATA_STUDIO_DATASET_VIEW_KEY, "Definition"))

        if dataset_view == "Definition":
            _render_section_heading("Definition", caption="This is the dataset currently loaded into Data Studio.")
            saved_message = st.session_state.pop(DATASET_SAVE_MESSAGE_KEY, None)
            if saved_message:
                st.success(saved_message)
            dataset["name"] = st.text_input(
                "Dataset name",
                value=str(dataset.get("name") or ""),
                key="data_studio.dataset.name",
            ).strip()
            dataset["description"] = st.text_area(
                "Description",
                value=str(dataset.get("description") or ""),
                key="data_studio.dataset.description",
            ).strip()

            source_options = list(SUPPORTED_DATASET_SOURCES)
            current_source = str(dataset.get("provider", {}).get("kind") or "synthetic")
            source = st.selectbox(
                "Source",
                options=source_options,
                index=source_options.index(current_source) if current_source in source_options else 0,
                format_func=lambda value: SOURCE_LABELS[value],
                key="data_studio.dataset.source",
            )
            dataset = switch_dataset_source(dataset, source)

            if source == "synthetic":
                _render_synthetic_dataset_editor(dataset)
            else:
                tabular_inspection = _render_tabular_dataset_editor(dataset)

            _sync_dataset_to_config(dataset)

            st.caption("Saving with an existing name updates that dataset in the catalog.")
            if st.button(
                "Save dataset",
                type="primary",
                use_container_width=True,
                disabled=not dataset.get("name"),
            ):
                try:
                    _save_dataset(dataset)
                except Exception as exc:
                    st.error(f"Could not save dataset: {type(exc).__name__}: {exc}")
                else:
                    st.rerun()

        preview_frame, preview_error = _build_preview_payload(dataset, tabular_inspection)

        if dataset_view == "Preview":
            _render_preview_tab(dataset, preview_frame, preview_error)

        if dataset_view == "Statistics":
            _render_statistics_tab(dataset, preview_frame, preview_error)
