
import base64
import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table
from dash.exceptions import PreventUpdate


# -----------------------------
# Parsing and preprocessing
# -----------------------------
REQUIRED_COLUMNS = ["DATE", "TIME", "SIM", "OBS"]


def parse_uploaded_csv(contents: str, filename: str, blank_obs_mode: str = "zero"):
    if not contents:
        raise ValueError("No file uploaded.")

    try:
        _, content_string = contents.split(",", 1)
    except ValueError as e:
        raise ValueError("Uploaded file format is invalid.") from e

    decoded = base64.b64decode(content_string)
    text_stream = io.StringIO(decoded.decode("utf-8-sig"))
    df = pd.read_csv(text_stream)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    df = df.copy()
    df["DATE"] = df["DATE"].astype(str).str.strip()
    df["TIME"] = df["TIME"].astype(str).str.strip()

    # Expected examples:
    # DATE = 02Oct2014
    # TIME = 05:00 or 17:00
    df["datetime"] = pd.to_datetime(
        df["DATE"] + " " + df["TIME"],
        format="%d%b%Y %H:%M",
        errors="coerce",
    )

    df["SIM"] = pd.to_numeric(df["SIM"], errors="coerce")
    obs_raw = pd.to_numeric(df["OBS"], errors="coerce")

    if blank_obs_mode == "zero":
        df["OBS"] = obs_raw.fillna(0.0)
    elif blank_obs_mode == "drop":
        df["OBS"] = obs_raw
    else:
        raise ValueError("Invalid blank OBS mode.")

    df = df.dropna(subset=["datetime", "SIM"])

    if blank_obs_mode == "drop":
        df = df.dropna(subset=["OBS"])

    df = df.sort_values("datetime").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows found after parsing the CSV.")

    return df


def infer_time_step_seconds(df: pd.DataFrame) -> float:
    deltas = df["datetime"].diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0]

    if deltas.empty:
        return float("nan")

    # Use the modal step first; if none, median is a reasonable fallback.
    modes = deltas.mode()
    if not modes.empty:
        return float(modes.iloc[0])
    return float(deltas.median())


def filter_by_xrange(df: pd.DataFrame, relayout_data: dict | None) -> pd.DataFrame:
    if df.empty:
        return df

    if not relayout_data:
        return df

    # Plotly sends different relayout payloads depending on the interaction.
    x0 = None
    x1 = None

    if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
        x0 = relayout_data["xaxis.range[0]"]
        x1 = relayout_data["xaxis.range[1]"]
    elif "xaxis.range" in relayout_data and isinstance(relayout_data["xaxis.range"], (list, tuple)) and len(relayout_data["xaxis.range"]) == 2:
        x0, x1 = relayout_data["xaxis.range"]
    elif relayout_data.get("xaxis.autorange"):
        return df

    if x0 is None or x1 is None:
        return df

    start = pd.to_datetime(x0, errors="coerce")
    end = pd.to_datetime(x1, errors="coerce")

    if pd.isna(start) or pd.isna(end):
        return df

    filtered = df[(df["datetime"] >= start) & (df["datetime"] <= end)].copy()
    return filtered if not filtered.empty else df.iloc[0:0].copy()


# -----------------------------
# HEC-HMS metrics
# -----------------------------
# Formulas implemented from the HEC-HMS documentation:
# NSE = 1 - sum((obs-sim)^2) / sum((obs-mean_obs)^2)
# RSR = sqrt(sum((obs-sim)^2)) / sqrt(sum((obs-mean_obs)^2))
# PBIAS = 100 * sum(sim-obs) / sum(obs)
# R^2 = squared Pearson correlation coefficient
# MKGE = 1 - sqrt((r-1)^2 + (beta-1)^2 + (gamma-1)^2)
# beta = mean(sim) / mean(obs)
# gamma = CV_sim / CV_obs


def nse_hec_hms(obs: np.ndarray, sim: np.ndarray):
    sse = np.sum((obs - sim) ** 2)
    sst = np.sum((obs - np.mean(obs)) ** 2)
    if sst == 0:
        return np.nan
    return 1.0 - (sse / sst)


def rsr_hec_hms(obs: np.ndarray, sim: np.ndarray):
    sse = np.sum((obs - sim) ** 2)
    sst = np.sum((obs - np.mean(obs)) ** 2)
    if sst == 0:
        return np.nan
    return np.sqrt(sse) / np.sqrt(sst)


def pbias_hec_hms(obs: np.ndarray, sim: np.ndarray):
    denom = np.sum(obs)
    if denom == 0:
        return np.nan
    return 100.0 * np.sum(sim - obs) / denom


def r2_hec_hms(obs: np.ndarray, sim: np.ndarray):
    if len(obs) < 2:
        return np.nan
    obs_std = np.std(obs, ddof=1)
    sim_std = np.std(sim, ddof=1)
    if obs_std == 0 or sim_std == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    return r ** 2


def mkge_hec_hms(obs: np.ndarray, sim: np.ndarray):
    if len(obs) < 2:
        return np.nan

    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)

    obs_std = np.std(obs, ddof=1)
    sim_std = np.std(sim, ddof=1)

    if mean_obs == 0 or obs_std == 0 or mean_sim == 0:
        return np.nan

    r = np.corrcoef(obs, sim)[0, 1]
    if np.isnan(r):
        return np.nan

    beta = mean_sim / mean_obs
    cv_obs = obs_std / mean_obs
    cv_sim = sim_std / mean_sim

    if cv_obs == 0:
        return np.nan

    gamma = cv_sim / cv_obs
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2)


def compute_metrics_and_volume(df_window: pd.DataFrame, dt_seconds: float):
    if df_window.empty:
        return {
            "NSE": np.nan,
            "RSR": np.nan,
            "PBIAS (%)": np.nan,
            "R²": np.nan,
            "Modified KGE": np.nan,
            "Observed Volume (m³)": np.nan,
            "Simulated Volume (m³)": np.nan,
            "Rows in Window": 0,
        }

    obs = df_window["OBS"].to_numpy(dtype=float)
    sim = df_window["SIM"].to_numpy(dtype=float)

    if math.isnan(dt_seconds) or dt_seconds <= 0:
        obs_vol = np.nan
        sim_vol = np.nan
    else:
        obs_vol = float(np.sum(obs * dt_seconds))
        sim_vol = float(np.sum(sim * dt_seconds))

    return {
        "NSE": nse_hec_hms(obs, sim),
        "RSR": rsr_hec_hms(obs, sim),
        "PBIAS (%)": pbias_hec_hms(obs, sim),
        "R²": r2_hec_hms(obs, sim),
        "Modified KGE": mkge_hec_hms(obs, sim),
        "Observed Volume (m³)": obs_vol,
        "Simulated Volume (m³)": sim_vol,
        "Rows in Window": len(df_window),
    }


def format_metric(value, digits=4):
    if pd.isna(value):
        return "NaN"
    return f"{value:.{digits}f}"


def format_volume(volume_m3):
    if pd.isna(volume_m3):
        return "NaN"
    if abs(volume_m3) >= 1_000_000:
        return f"{volume_m3 / 1000:.2f} x10³ m³"
    return f"{volume_m3:.2f} m³"


def make_metrics_rows(metrics: dict, visible_start=None, visible_end=None):
    rows = [
        {"Metric": "Visible Start", "Value": str(visible_start) if visible_start is not None else "-"},
        {"Metric": "Visible End", "Value": str(visible_end) if visible_end is not None else "-"},
        {"Metric": "Rows in Window", "Value": metrics.get("Rows in Window", 0)},
        {"Metric": "NSE", "Value": format_metric(metrics["NSE"])},
        {"Metric": "RSR", "Value": format_metric(metrics["RSR"])},
        {"Metric": "PBIAS (%)", "Value": format_metric(metrics["PBIAS (%)"])},
        {"Metric": "R²", "Value": format_metric(metrics["R²"])},
        {"Metric": "Modified KGE", "Value": format_metric(metrics["Modified KGE"])},
        {"Metric": "Observed Volume", "Value": format_volume(metrics["Observed Volume (m³)"])},
        {"Metric": "Simulated Volume", "Value": format_volume(metrics["Simulated Volume (m³)"])},
    ]
    return rows


def make_figure(df: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["OBS"],
            mode="lines",
            name="Observed Flow",
            line={"width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["SIM"],
            mode="lines",
            name="Simulated Flow",
            line={"width": 2},
        )
    )

    fig.update_layout(
        title="Observed and Simulated Hydrograph",
        margin={"l": 60, "r": 30, "t": 50, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        xaxis_title="Date-Time",
        yaxis_title="Flow",
        template="plotly_white",
        hovermode="x unified",
        uirevision="keep-zoom",
    )

    return fig


# -----------------------------
# App
# -----------------------------
app = Dash(__name__)
app.title = "HEC-HMS Metrics Viewer"

app.layout = html.Div(
    style={
        "height": "100vh",
        "display": "flex",
        "flexDirection": "column",
        "fontFamily": "Arial, sans-serif",
        "padding": "12px",
        "boxSizing": "border-box",
        "gap": "12px",
    },
    children=[
        dcc.Store(id="stored-data"),
        dcc.Store(id="stored-time-step"),
        html.Div(
            style={
                "flex": "0 0 42%",
                "minHeight": "280px",
                "border": "1px solid #d9d9d9",
                "borderRadius": "10px",
                "padding": "12px",
                "overflowY": "auto",
                "boxSizing": "border-box",
                "backgroundColor": "#fafafa",
            },
            children=[
                html.H2("HEC-HMS Goodness-of-Fit Viewer", style={"marginTop": "0"}),
                html.Div(
                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "center"},
                    children=[
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(["Drag and drop CSV here or ", html.B("click to upload")]),
                            style={
                                "width": "360px",
                                "height": "72px",
                                "lineHeight": "72px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "8px",
                                "textAlign": "center",
                                "backgroundColor": "white",
                            },
                            multiple=False,
                        ),
                        html.Div(
                            style={"minWidth": "240px"},
                            children=[
                                html.Div("Blank OBS handling", style={"fontWeight": "bold", "marginBottom": "6px"}),
                                dcc.RadioItems(
                                    id="blank-obs-mode",
                                    options=[
                                        {"label": "Treat blank OBS as 0", "value": "zero"},
                                        {"label": "Ignore rows with blank OBS", "value": "drop"},
                                    ],
                                    value="zero",
                                    labelStyle={"display": "block", "marginBottom": "4px"},
                                ),
                            ],
                        ),
                        html.Div(
                            id="file-status",
                            style={"fontSize": "14px", "color": "#333"},
                            children="Upload a CSV file with DATE, TIME, SIM, OBS.",
                        ),
                    ],
                ),
                html.Hr(),
                html.Div(
                    style={"marginBottom": "8px", "fontSize": "14px"},
                    children=[
                        html.B("How metrics are computed: "),
                        "The metrics update from the visible x-axis interval only. Zoom or pan the hydrograph below to change the calculation window."
                    ],
                ),
                dash_table.DataTable(
                    id="metrics-table",
                    columns=[{"name": "Metric", "id": "Metric"}, {"name": "Value", "id": "Value"}],
                    data=[],
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "padding": "8px",
                        "fontSize": "14px",
                        "whiteSpace": "normal",
                        "height": "auto",
                    },
                    style_header={"fontWeight": "bold"},
                ),
            ],
        ),
        html.Div(
            style={
                "flex": "1 1 58%",
                "minHeight": "320px",
                "border": "1px solid #d9d9d9",
                "borderRadius": "10px",
                "padding": "6px",
                "boxSizing": "border-box",
                "backgroundColor": "white",
            },
            children=[
                dcc.Graph(
                    id="hydrograph",
                    figure=go.Figure(),
                    style={"height": "100%"},
                    config={
                        "displayModeBar": True,
                        "scrollZoom": True,
                        "responsive": True,
                    },
                )
            ],
        ),
    ],
)


@app.callback(
    Output("stored-data", "data"),
    Output("stored-time-step", "data"),
    Output("hydrograph", "figure"),
    Output("file-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("blank-obs-mode", "value"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename, blank_obs_mode):
    if not contents:
        raise PreventUpdate

    try:
        df = parse_uploaded_csv(contents, filename, blank_obs_mode=blank_obs_mode)
        dt_seconds = infer_time_step_seconds(df)
        fig = make_figure(df)

        status = (
            f"Loaded {filename} | Rows: {len(df)} | "
            f"Start: {df['datetime'].min()} | End: {df['datetime'].max()} | "
            f"Time step used for runoff volume: {dt_seconds:.0f} seconds"
            if not math.isnan(dt_seconds)
            else f"Loaded {filename} | Rows: {len(df)} | Time step could not be inferred."
        )

        data_json = df.to_json(date_format="iso", orient="split")
        return data_json, dt_seconds, fig, status

    except Exception as e:
        empty_fig = go.Figure()
        return None, None, empty_fig, f"Error: {e}"


@app.callback(
    Output("metrics-table", "data"),
    Input("stored-data", "data"),
    Input("stored-time-step", "data"),
    Input("hydrograph", "relayoutData"),
)
def update_metrics(data_json, dt_seconds, relayout_data):
    if not data_json:
        return []

    df = pd.read_json(io.StringIO(data_json), orient="split")
    df["datetime"] = pd.to_datetime(df["datetime"])

    df_window = filter_by_xrange(df, relayout_data)

    if df_window.empty:
        return [{"Metric": "Status", "Value": "No rows fall inside the current visible x-axis interval."}]

    metrics = compute_metrics_and_volume(df_window, float(dt_seconds) if dt_seconds is not None else float("nan"))

    visible_start = df_window["datetime"].min()
    visible_end = df_window["datetime"].max()

    return make_metrics_rows(metrics, visible_start, visible_end)


if __name__ == "__main__":
    app.run(debug=True)
