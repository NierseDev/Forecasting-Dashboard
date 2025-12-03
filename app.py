# Standard library
import re
from datetime import date
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(
    page_title="Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Forecasting Dashboard")

# --- 1. Dataset/s Loading ---
@st.cache_data
def load_datasets(data_dir="DATA", extension=".csv"):

    data_dir = Path(data_dir)
    datasets = {}

    if not data_dir.exists() or not data_dir.is_dir():
        return datasets

    ext = extension if extension.startswith(".") else f".{extension}"
    for f in sorted(data_dir.glob(f"*{ext}")):
        try:
            df = pd.read_csv(f)
        except Exception:
            try:
                df = pd.read_csv(f, encoding="utf-8", engine="python")
            except Exception:
                # skip files that cannot be read as CSV
                continue

        try:
            # Normalize column names to upper-case
            df.columns = [c.strip().upper() for c in df.columns]

            # Ensure DATE/YEAR/MONTH/DAY are present and consistent
            if "DATE" in df.columns:
                df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
                df["YEAR"] = df["DATE"].dt.year
                df["MONTH"] = df["DATE"].dt.month
                df["DAY"] = df["DATE"].dt.day
            elif {"YEAR", "MONTH", "DAY"}.issubset(set(df.columns)):
                df["DATE"] = pd.to_datetime(
                    {"year": df["YEAR"], "month": df["MONTH"], "day": df["DAY"]},
                    errors="coerce",
                )

            # Capture seasonality (simplified cyclical encodings)
            if "DATE" in df.columns:

                doy = df["DATE"].dt.dayofyear
                df["DAY_OF_YEAR"] = doy
                df["DOY_SIN"] = np.sin(2 * np.pi * doy / 365.25)
                df["DOY_COS"] = np.cos(2 * np.pi * doy / 365.25)

                m = df["DATE"].dt.month
                df["MONTH_SIN"] = np.sin(2 * np.pi * m / 12)
                df["MONTH_COS"] = np.cos(2 * np.pi * m / 12)
            else:
                # fallback: if YEAR/MONTH/DAY exist, build DATE then seasonal features
                if {"YEAR", "MONTH", "DAY"}.issubset(set(df.columns)):
                    df["DATE"] = pd.to_datetime(
                        {"year": df["YEAR"], "month": df["MONTH"], "day": df["DAY"]},
                        errors="coerce",
                    )

                    doy = df["DATE"].dt.dayofyear
                    df["DAY_OF_YEAR"] = doy
                    df["DOY_SIN"] = np.sin(2 * np.pi * doy / 365.25)
                    df["DOY_COS"] = np.cos(2 * np.pi * doy / 365.25)

                    m = df["DATE"].dt.month
                    df["MONTH_SIN"] = np.sin(2 * np.pi * m / 12)
                    df["MONTH_COS"] = np.cos(2 * np.pi * m / 12)

            # Standardize and encode wind direction (circular variable)
            wdir_candidates = ["WIND_DIRECTION", "WIND_DIR", "WINDDEG", "WINDDEGREES"]
            # 16-point compass mapping to degrees
            compass_map = {
                "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
                "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
                "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
                "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
            }

            def _parse_wind_dir_series(s):
                # Return numpy array of degrees (float) with NaNs where unparsable
                if s.dtype.kind in "iuf":  # numeric
                    arr = s.astype(float).to_numpy()
                else:
                    # try to parse strings: compass points or numeric embedded
                    parsed = []
                    for v in s.astype(str).fillna("").str.strip().str.upper().to_list():
                        if not v:
                            parsed.append(np.nan)
                            continue
                        # remove degree symbol and common words
                        v_clean = v.replace("°", "").replace("DEG", "").replace("DEGREES", "").strip()
                        # direct compass map
                        if v_clean in compass_map:
                            parsed.append(compass_map[v_clean])
                            continue
                        # sometimes values include multiple tokens, take first token
                        token = v_clean.split()[0]
                        # try compass again
                        if token in compass_map:
                            parsed.append(compass_map[token])
                            continue
                        # extract numeric
                        m = re.search(r"[-+]?\d+(\.\d+)?", token)
                        if m:
                            try:
                                parsed.append(float(m.group(0)))
                            except Exception:
                                parsed.append(np.nan)
                        else:
                            parsed.append(np.nan)
                    arr = np.array(parsed, dtype=float)
                # normalize to [0,360)
                with np.errstate(invalid="ignore"):
                    arr = np.mod(arr, 360.0)
                    arr = np.where(np.isfinite(arr), arr, np.nan)
                return arr

            for cand in wdir_candidates:
                if cand in df.columns:
                    try:
                        degs = _parse_wind_dir_series(df[cand])
                        # overwrite canonical column to be numeric degrees
                        df[cand] = degs
                        # add circular encodings as features for modeling
                        radians = np.deg2rad(degs)
                        df[cand + "_SIN"] = np.where(np.isnan(degs), np.nan, np.sin(radians))
                        df[cand + "_COS"] = np.where(np.isnan(degs), np.nan, np.cos(radians))
                    except Exception:
                        # if something goes wrong, leave original column as-is
                        pass

            # Set CITY from filename (first token before '_'), ensure proper noun formatting
            city = f.stem.split("_")[0]
            city = city.replace("-", " ").strip().title()
            df["CITY"] = city

            datasets[f.stem] = df
        except Exception:
            # on unexpected processing error, keep raw dataframe
            datasets[f.stem] = df

    return datasets

# --- 2. Model Training ---
@st.cache_resource
def train_forecast_model(df):
    # train up to 2024-12-31, validate with the rest
    CUTOFF = pd.Timestamp("2024-12-31")

    def _angle_rmse_deg(pred_deg, true_deg):
        # minimal signed difference in degrees, then RMSE
        diff = ((pred_deg - true_deg + 180.0) % 360.0) - 180.0
        return float(np.sqrt(np.mean(np.square(diff))))

    def _rmse(a, b):
        return float(np.sqrt(np.mean(np.square(a - b))))

    def _train_for_df(df):
        df = df.copy()
        colset = set(df.columns)

        # ensure DATE exists as datetime if present
        if "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

        # prioritize common target column names (upper-case expected)
        rain_candidates = ["RAINFALL", "RAIN", "PRECIPITATION", "PRECIP"]
        wdir_candidates = ["WIND_DIRECTION", "WIND_DIR", "WINDDEG", "WINDDEGREES"]
        wspd_candidates = ["WIND_SPEED", "WINDSPD", "WIND"]

        def _find(cands):
            for c in cands:
                if c in colset:
                    return c
            return None

        targets = {
            "rainfall": _find(rain_candidates),
            "wind_direction": _find(wdir_candidates),
            "wind_speed": _find(wspd_candidates),
        }

        # preferred feature set (seasonal / time features added during loading)
        pref_feats = ["DOY_SIN", "DOY_COS", "MONTH_SIN", "MONTH_COS", "DAY_OF_YEAR", "YEAR", "MONTH", "DAY"]
        features = [f for f in pref_feats if f in colset]

        # fallback: use any numeric columns except target columns and identifiers
        if not features:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            excluded = set(v for v in targets.values() if v)
            excluded.update({"DAY_OF_YEAR"})
            features = [c for c in numeric if c not in excluded]

        models = {}
        for logical_name, col in targets.items():
            if col is None or col not in df.columns:
                continue

            use_cols = [c for c in features if c in df.columns]
            if not use_cols:
                continue

            # split by date if available
            if "DATE" in df.columns:
                train_mask = df["DATE"] <= CUTOFF
                val_mask = df["DATE"] > CUTOFF
            else:
                # no date -> keep previous behaviour (train on all, no validation)
                train_mask = pd.Series(True, index=df.index)
                val_mask = pd.Series(False, index=df.index)

            # Special handling for circular wind direction: train on SIN/COS as multivariate target if available
            if logical_name == "wind_direction":
                sin_col = col + "_SIN"
                cos_col = col + "_COS"
                if sin_col in df.columns and cos_col in df.columns:
                    train_df = df.loc[train_mask, use_cols + [sin_col, cos_col]].dropna()
                    if train_df.shape[0] < 5:
                        continue
                    X_train = train_df[use_cols].values
                    y_train = train_df[[sin_col, cos_col]].values
                    model = LinearRegression()
                    model.fit(X_train, y_train)

                    info = {
                        "model": model,
                        "features": use_cols,
                        "target_column": col,
                        "circular": True,
                        "target_columns": [sin_col, cos_col],
                    }

                    # validation if available
                    val_df = df.loc[val_mask, use_cols + [col, sin_col, cos_col]].dropna(subset=use_cols + [col])
                    if val_df.shape[0] > 0:
                        X_val = val_df[use_cols].values
                        y_val_deg = val_df[col].values  # degrees ground truth
                        y_pred_sincos = model.predict(X_val)
                        # convert predicted sin/cos to degrees
                        pred_rad = np.arctan2(y_pred_sincos[:, 0], y_pred_sincos[:, 1])
                        pred_deg = (np.degrees(pred_rad) + 360.0) % 360.0
                        rmse_angle = _angle_rmse_deg(pred_deg, y_val_deg)
                        info["validation"] = {"n_train": int(X_train.shape[0]), "n_val": int(X_val.shape[0]), "rmse_angle_deg": rmse_angle}
                    else:
                        info["validation"] = {"n_train": int(X_train.shape[0]), "n_val": 0}

                    models[logical_name] = info
                    continue
                # if sin/cos not available, fall back to predicting angle directly

            # default (non-circular or fallback) behavior: train on train split
            train_df = df.loc[train_mask, use_cols + [col]].dropna()
            if train_df.shape[0] < 5:
                continue

            # Special-case: ensure rainfall units are interpreted consistently (expect input in mm).
            unit_info = {"input_unit": None, "model_unit": "mm", "applied_scale": 1.0}
            if logical_name == "rainfall":
                # inspect typical magnitudes to guess units
                vals = train_df[col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    med = float(np.median(vals))
                    mx = float(np.max(vals))
                    # heuristics:
                    # - if values look like meters (e.g., med < 1 and max < 10), convert meters -> millimeters
                    # - otherwise assume already in millimeters
                    if med < 1.0 and mx < 10.0:
                        unit_info["input_unit"] = "m"
                        unit_info["applied_scale"] = 1000.0
                    else:
                        unit_info["input_unit"] = "mm"
                        unit_info["applied_scale"] = 1.0
                else:
                    unit_info["input_unit"] = "unknown"
                    unit_info["applied_scale"] = 1.0

                if unit_info["applied_scale"] != 1.0:
                    # apply scaling in-place on copies used for training/validation
                    train_df[col] = train_df[col].astype(float) * unit_info["applied_scale"]

            X_train = train_df[use_cols].values
            y_train = train_df[col].values

            model = LinearRegression()
            model.fit(X_train, y_train)

            info = {"model": model, "features": use_cols, "target_column": col, "circular": False, "units": unit_info}

            # validation if possible
            val_df = df.loc[val_mask, use_cols + [col]].dropna()
            if val_df.shape[0] > 0:
                # apply same rainfall scaling to validation target if applicable
                if logical_name == "rainfall" and unit_info["applied_scale"] != 1.0:
                    val_df[col] = val_df[col].astype(float) * unit_info["applied_scale"]

                X_val = val_df[use_cols].values
                y_val = val_df[col].values
                y_pred = model.predict(X_val)

                if logical_name == "wind_direction":
                    # treat as circular angle prediction
                    rmse_angle = _angle_rmse_deg(y_pred, y_val)
                    info["validation"] = {"n_train": int(X_train.shape[0]), "n_val": int(X_val.shape[0]), "rmse_angle_deg": rmse_angle}
                else:
                    rmse = _rmse(y_pred, y_val)
                    info["validation"] = {"n_train": int(X_train.shape[0]), "n_val": int(X_val.shape[0]), "rmse": rmse}
            else:
                info["validation"] = {"n_train": int(X_train.shape[0]), "n_val": 0}

            models[logical_name] = info

        return models

    # main entry: accept either a single DataFrame or a dict of DataFrames
    if isinstance(df, dict):
        trained = {}
        for k, v in df.items():
            try:
                trained[k] = _train_for_df(v)
            except Exception:
                trained[k] = {}
        return trained
    else:
        try:
            return _train_for_df(df)
        except Exception:
            return {}
        
# --- 3. User Interface ---
# load available datasets
datasets = load_datasets("DATA")
# collect city names from loaded DataFrames
cities = sorted({df["CITY"].dropna().iloc[0] for df in datasets.values() if "CITY" in df.columns and len(df["CITY"].dropna()) > 0})

# --- Sidebar configuration ---
st.sidebar.header("Configuration")

# Location selection
loc_mode = st.sidebar.radio("Location selection", ("All", "By City"))
selected_city = None
if loc_mode == "By City":
    if cities:
        selected_city = st.sidebar.selectbox("Choose city", [""] + cities)
        if selected_city == "":
            selected_city = None
    else:
        st.sidebar.info("No city metadata available in DATA folder.")

# What to forecast
forecast_choice = st.sidebar.radio(
    "What to forecast",
    ("Rainfall", "Wind (speed & direction)", "All (combined)")
)

# Forecast duration and aggregation
# Forecast duration / aggregation / date-format selection
agg = st.sidebar.radio("Forecast aggregation", ("By day", "By month"))

# allowed range: 2025-01-01 .. 2030-12-31
_min_date = date(2025, 1, 1)
_max_date = date(2030, 12, 31)

if agg == "By month":
    st.sidebar.write("Select start/end months. Format: YYYY-MM (e.g. 2025-2).")
    start_picker = st.sidebar.date_input("Start month", value=_min_date, min_value=_min_date, max_value=_max_date, help="Day is ignored; the first of the month will be used.")
    end_picker = st.sidebar.date_input("End month", value=_max_date, min_value=_min_date, max_value=_max_date, help="Day is ignored; the last day of the month will be used.")

    # normalize to month boundaries
    start_date = start_picker.replace(day=1)
    end_month_ts = pd.to_datetime(end_picker)
    end_date = (end_month_ts + pd.offsets.MonthEnd(0)).date()

    # readable examples without zero-padding (matches requested examples like '2025-2' or '2028-10')
    start_example = f"{start_date.year}-{start_date.month}"
    end_example = f"{end_date.year}-{end_date.month}"

else:  # By day
    st.sidebar.write("Select exact start/end dates. Format: YYYY-MM-DD\n(e.g. 2025-10-13).")
    start_date = st.sidebar.date_input("Start date", value=_min_date, min_value=_min_date, max_value=_max_date)
    end_date = st.sidebar.date_input("End date", value=_max_date, min_value=_min_date, max_value=_max_date)

    # examples without zero-padding
    start_example = f"{start_date.year}-{start_date.month}-{start_date.day}"
    end_example = f"{end_date.year}-{end_date.month}-{end_date.day}"

# ensure sensible ordering
if start_date > end_date:
    st.sidebar.warning("Start is after end — swapping the two dates.")
    start_date, end_date = end_date, start_date

# Run action
run = st.sidebar.button("Run forecast")

if run:
    st.sidebar.success("Running forecast...")
    if loc_mode == "By City" and not selected_city:
        st.warning("Please select a city when using 'By City' location selection.")
    else:
        # pick datasets according to selection
        if loc_mode == "All":
            chosen = datasets
        else:
            chosen = {
                k: v for k, v in datasets.items()
                if "CITY" in v.columns and v["CITY"].dropna().iloc[0] == selected_city
            }
        if not chosen:
            st.warning("No datasets matched the selection.")
        else:
            # determine which logical targets to produce based on user's choice
            if forecast_choice == "Rainfall":
                allowed_targets = {"rainfall"}
            elif forecast_choice == "Wind (speed & direction)":
                allowed_targets = {"wind_speed", "wind_direction"}
            else:  # "All (combined)"
                allowed_targets = {"rainfall", "wind_speed", "wind_direction"}

            with st.spinner("Training models on selected dataset(s)..."):
                trained = train_forecast_model(chosen)

            # build forecast index according to aggregation choice
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)

            freq = "MS" if agg == "By month" else "D"
            forecast_index = pd.date_range(start=start_ts, end=end_ts, freq=freq)

            for name, df in chosen.items():
                st.header(f"Dataset: {name}")
                models = trained.get(name, {})
                if not models:
                    st.info("No model could be trained for this dataset.")
                    continue

                # prepare future frame with canonical time features expected by models
                future = pd.DataFrame({"DATE": forecast_index})
                future["YEAR"] = future["DATE"].dt.year
                future["MONTH"] = future["DATE"].dt.month
                future["DAY"] = future["DATE"].dt.day
                doy = future["DATE"].dt.dayofyear
                future["DAY_OF_YEAR"] = doy
                future["DOY_SIN"] = np.sin(2 * np.pi * doy / 365.25)
                future["DOY_COS"] = np.cos(2 * np.pi * doy / 365.25)
                m = future["DATE"].dt.month
                future["MONTH_SIN"] = np.sin(2 * np.pi * m / 12)
                future["MONTH_COS"] = np.cos(2 * np.pi * m / 12)

                # container for predicted series
                preds = pd.DataFrame({"DATE": future["DATE"]})

                for logical_name, info in models.items():
                    # honor user's "What to forecast" choice
                    if logical_name not in allowed_targets:
                        continue

                    model = info.get("model")
                    feats = info.get("features", [])
                    if model is None or not feats:
                        continue

                    X = future.reindex(columns=feats).copy()
                    # Replace any remaining NaNs in features with zeros (simple strategy)
                    X = X.fillna(0).values

                    try:
                        if info.get("circular", False):
                            y_sincos = model.predict(X)
                            # convert predicted sin/cos to degrees (atan2(sin, cos))
                            pred_rad = np.arctan2(y_sincos[:, 0], y_sincos[:, 1])
                            pred_deg = (np.degrees(pred_rad) + 360.0) % 360.0
                            col_name = "PRED_" + (logical_name.upper())
                            preds[col_name] = pred_deg
                        else:
                            y_pred = model.predict(X)
                            col_name = "PRED_" + (logical_name.upper())
                            # for rainfall keep model units (info["units"]["model_unit"])
                            preds[col_name] = y_pred
                    except Exception as e:
                        st.warning(f"Prediction failed for {name} / {logical_name}: {e}")
                        continue

                preds = preds.set_index("DATE")

                # Prepare actuals (if available) and limit to user-specified end date
                actuals = pd.DataFrame(index=forecast_index)
                if "DATE" in df.columns:
                    df_local = df.copy()
                    df_local["DATE"] = pd.to_datetime(df_local["DATE"], errors="coerce")
                    # filter actuals to not exceed end_ts (requirement)
                    df_local = df_local[df_local["DATE"] <= end_ts]
                    actual_idx = df_local["DATE"]
                    # generic: attach known target columns if present
                    for targ_key in ("RAINFALL", "RAIN", "PRECIPITATION", "PRECIP",
                                     "WIND_SPEED", "WINDSPD", "WIND",
                                     "WIND_DIRECTION", "WIND_DIR", "WINDDEG", "WINDDEGREES"):
                        # only attach actuals that match the allowed forecast types
                        if targ_key in ("RAINFALL", "RAIN", "PRECIPITATION", "PRECIP"):
                            if "rainfall" not in allowed_targets:
                                continue
                        if targ_key in ("WIND_SPEED", "WINDSPD", "WIND"):
                            if "wind_speed" not in allowed_targets:
                                continue
                        if targ_key in ("WIND_DIRECTION", "WIND_DIR", "WINDDEG", "WINDDEGREES"):
                            if "wind_direction" not in allowed_targets:
                                continue

                        if targ_key in df_local.columns:
                            s = df_local.set_index("DATE")[targ_key].rename(f"ACT_{targ_key}")
                            # align to forecast_index (will introduce NaNs where missing)
                            actuals = actuals.join(s, how="left")

                actuals = actuals.set_index(forecast_index)

                # Merge preds and actuals for plotting
                to_plot = preds.join(actuals, how="outer").loc[:end_ts]  # ensure not beyond end_ts

                if to_plot.empty:
                    st.info("No forecast/actual data available to plot for this dataset in the requested range.")
                    continue

                st.subheader("Forecast vs Actual (limited to selected end date)")

                # group columns by logical variable to produce clearer plotly charts
                cols = list(to_plot.columns)

                rain_cols = [c for c in cols if any(k in c for k in ["RAIN", "PRECIP"]) and "rainfall" in allowed_targets]
                dir_cols = [c for c in cols if any(k in c for k in ["DIRECTION", "WIND_DIR", "WINDDEG", "WINDDEGREES", "WIND_DIRECTION"]) and "wind_direction" in allowed_targets]
                # wind speed: columns mentioning WIND and not in direction set, or explicit WIND_SPEED/WINDSPD
                spd_cols = [c for c in cols if (("WIND_SPEED" in c or "WINDSPD" in c) or ("WIND" in c and c not in dir_cols and "SPEED" not in c and "DIR" not in c and "DIRECTION" not in c)) and "wind_speed" in allowed_targets]

                # helper to plot a group if non-empty
                def _plot_group(group_cols, title, yaxis_title=None):
                    if not group_cols:
                        return
                    df_plot = to_plot.reset_index().rename(columns={"index": "DATE"})
                    fig = px.line(df_plot, x="DATE", y=group_cols, title=title)
                    # style predicted series as dashed (names containing 'PRED') to distinguish from actuals
                    for i, tr in enumerate(fig.data):
                        if "PRED" in (tr.name or "").upper():
                            # apply dash style on the line attribute for Scatter/Scattergl traces
                            fig.data[i].update(line=dict(dash="dash"))
                        # add markers for clarity
                        fig.data[i].update(mode="lines+markers")
                    if yaxis_title:
                        fig.update_yaxes(title_text=yaxis_title)
                    st.plotly_chart(fig, use_container_width=True)

                if "rainfall" in allowed_targets:
                    _plot_group(rain_cols, f"{name} — Rainfall (predicted vs actual)", "Rain (model units)")
                if "wind_speed" in allowed_targets:
                    _plot_group(spd_cols, f"{name} — Wind Speed (predicted vs actual)", "Wind speed")
                if "wind_direction" in allowed_targets:
                    _plot_group(dir_cols, f"{name} — Wind Direction (predicted vs actual)", "Degrees")

                # Show numeric table of predictions
                st.subheader("Forecast table")
                display_df = preds.reset_index()

                try:
                    # try non-zero-padded form where supported
                    display_df["DATE_STR"] = display_df["DATE"].dt.strftime("%Y-%-m-%-d")
                except Exception:
                    display_df["DATE_STR"] = display_df["DATE"].dt.strftime("%Y-%m-%d")

                st.dataframe(display_df.set_index("DATE").head(1000))

            st.success("Forecast run complete.")