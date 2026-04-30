"""
Movie Success Predictor Dashboard (Dash + Plotly)

Run locally:
    pip install dash plotly pandas numpy scikit-learn
    python movie_dashboard.py

Run in Google Colab:
    !pip install dash jupyter-dash plotly scikit-learn
    # Make sure your cleaned CSV exists at:
    # /content/drive/MyDrive/cis2450_movie_project/movies_clean_modeling.csv
    !python movie_dashboard.py

Then open http://127.0.0.1:8050/ locally.

Expected data file:
    movies_clean_modeling.csv from the final project notebook.
You can override the path with:
    export MOVIE_DATA_PATH="/path/to/movies_clean_modeling.csv"
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False

RANDOM_STATE = 42
DEFAULT_PATHS = [
    Path(os.environ.get("MOVIE_DATA_PATH", "")),
    Path("/content/drive/MyDrive/cis2450_movie_project/movies_clean_modeling.csv"),
    Path("movies_clean_modeling.csv"),
    Path("movie_project_data/movies_clean_modeling.csv"),
]

FEATURES_NUMERIC = [
    "log_budget", "budget_missing", "runtime", "log_popularity", "vote_average",
    "log_vote_count", "release_year", "release_month", "genre_count",
    "production_company_count", "production_country_count", "spoken_language_count",
    "belongs_to_collection", "cast_size", "crew_size", "director_count",
    "writer_count", "producer_count", "keyword_count", "top10_cast_avg_popularity",
    "top10_cast_max_popularity", "overview_len", "title_len", "wikidata_matched",
    "wikidata_sitelinks", "wd_country_count", "wd_award_nomination_count",
    "wd_narrative_location_count", "wd_based_on_count",
]
FEATURES_CATEGORICAL = ["primary_genre", "original_language", "release_decade"]

DISPLAY_COLS = [
    "title", "release_year", "primary_genre", "original_language", "runtime",
    "budget", "revenue", "vote_average", "vote_count", "popularity", "blockbuster",
]

COLOR_PRIMARY = "#4f46e5"
COLOR_ACCENT = "#06b6d4"
COLOR_SUCCESS = "#16a34a"
COLOR_WARNING = "#f59e0b"
COLOR_DANGER = "#dc2626"
BG = "#f6f7fb"
CARD = "#ffffff"
TEXT = "#111827"
MUTED = "#6b7280"
BORDER = "#e5e7eb"


def find_data_path() -> Path:
    for p in DEFAULT_PATHS:
        if str(p) and p.exists():
            return p
    checked = "\n".join(f"- {p}" for p in DEFAULT_PATHS if str(p))
    raise FileNotFoundError(
        "Could not find movies_clean_modeling.csv. Checked:\n"
        f"{checked}\n\nSet MOVIE_DATA_PATH or place the file next to this script."
    )


def money(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    x = float(x)
    if abs(x) >= 1e9:
        return f"${x/1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:.1f}K"
    return f"${x:,.0f}"


def load_data() -> pd.DataFrame:
    path = find_data_path()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Normalize expected columns.
    if "name" in df.columns and "title" not in df.columns:
        df = df.rename(columns={"name": "title"})

    numeric_cols = set(FEATURES_NUMERIC + ["revenue", "budget", "runtime", "vote_average", "vote_count", "popularity", "release_year", "release_month", "log_revenue", "blockbuster"])
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "blockbuster" in df.columns:
        df["blockbuster"] = df["blockbuster"].fillna(0).astype(int)
    else:
        threshold = df["revenue"].quantile(0.80)
        df["blockbuster"] = (df["revenue"] >= threshold).astype(int)

    # Derived columns for dashboard if absent.
    if "log_revenue" not in df.columns and "revenue" in df.columns:
        df["log_revenue"] = np.log1p(df["revenue"].clip(lower=0))
    if "log_budget" not in df.columns and "budget" in df.columns:
        budget = df["budget"].where(df["budget"] > 0)
        df["log_budget"] = np.log1p(budget)
    if "log_popularity" not in df.columns and "popularity" in df.columns:
        df["log_popularity"] = np.log1p(df["popularity"].clip(lower=0))
    if "log_vote_count" not in df.columns and "vote_count" in df.columns:
        df["log_vote_count"] = np.log1p(df["vote_count"].clip(lower=0))
    if "budget_missing" not in df.columns:
        df["budget_missing"] = (df.get("budget", pd.Series(index=df.index, dtype=float)).fillna(0) <= 0).astype(int)
    if "release_decade" not in df.columns and "release_year" in df.columns:
        df["release_decade"] = ((df["release_year"] // 10) * 10).astype("Int64").astype(str)

    for c in FEATURES_CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)

    if "title" not in df.columns:
        df["title"] = "Movie " + df.index.astype(str)

    return df


def make_preprocess(df: pd.DataFrame) -> Tuple[List[str], List[str], ColumnTransformer]:
    num = [c for c in FEATURES_NUMERIC if c in df.columns]
    cat = [c for c in FEATURES_CATEGORICAL if c in df.columns]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=25)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_transformer, num),
        ("cat", categorical_transformer, cat),
    ], remainder="drop")
    return num, cat, preprocess


def fit_models(df: pd.DataFrame) -> Dict[str, object]:
    num, cat, preprocess = make_preprocess(df)
    features = num + cat
    modeling_df = df.dropna(subset=["log_revenue", "blockbuster"]).copy()
    X = modeling_df[features]
    y_reg = modeling_df["log_revenue"]
    y_clf = modeling_df["blockbuster"].astype(int)

    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
    )

    reg = Pipeline([
        ("preprocess", preprocess),
        ("model", HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=RANDOM_STATE)),
    ])
    reg.fit(X_train, yr_train)
    reg_pred = reg.predict(X_test)

    # Use the tuned RF + SMOTE setup if imblearn is available; otherwise fall back to HGB classifier.
    if HAS_IMBLEARN:
        clf = ImbPipeline([
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", RandomForestClassifier(
                n_estimators=300,
                min_samples_split=10,
                max_depth=None,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ])
        clf_name = "Tuned Random Forest + SMOTE"
    else:
        clf = Pipeline([
            ("preprocess", preprocess),
            ("model", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=RANDOM_STATE)),
        ])
        clf_name = "Histogram Gradient Boosting"

    clf.fit(X_train, yc_train)
    clf_pred = clf.predict(X_test)
    clf_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "reg_rmse": float(math.sqrt(mean_squared_error(yr_test, reg_pred))),
        "reg_r2": float(r2_score(yr_test, reg_pred)),
        "clf_f1": float(f1_score(yc_test, clf_pred)),
        "clf_auc": float(roc_auc_score(yc_test, clf_prob)),
        "clf_name": clf_name,
        "feature_count": len(features),
    }

    return {
        "features": features,
        "numeric_features": num,
        "categorical_features": cat,
        "reg_model": reg,
        "clf_model": clf,
        "metrics": metrics,
        "modeling_df": modeling_df,
    }


def apply_filters(df: pd.DataFrame, genres, year_range, revenue_range, only_blockbusters) -> pd.DataFrame:
    out = df.copy()
    if genres:
        out = out[out["primary_genre"].isin(genres)]
    if year_range and "release_year" in out.columns:
        out = out[out["release_year"].between(year_range[0], year_range[1], inclusive="both")]
    if revenue_range and "revenue" in out.columns:
        lo, hi = revenue_range
        out = out[out["revenue"].between(lo, hi, inclusive="both")]
    if only_blockbusters:
        out = out[out["blockbuster"] == 1]
    return out


def kpi_card(label: str, value: str, sub: str = ""):
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div(sub, className="kpi-sub"),
    ], className="kpi-card")


def section_title(title: str, subtitle: str = ""):
    return html.Div([
        html.H2(title, className="section-title"),
        html.Div(subtitle, className="section-subtitle") if subtitle else None,
    ])


def make_app() -> Dash:
    df = load_data()
    assets = fit_models(df)
    metrics = assets["metrics"]

    genres = sorted(df["primary_genre"].dropna().astype(str).unique()) if "primary_genre" in df else []
    year_min = int(np.nanmin(df["release_year"])) if "release_year" in df else 1950
    year_max = int(np.nanmax(df["release_year"])) if "release_year" in df else 2026
    rev_max = float(np.nanpercentile(df["revenue"].fillna(0), 99.5)) if "revenue" in df else 1e8

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Movie Success Predictor"

    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body { margin:0; background:#f6f7fb; font-family:Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color:#111827; }
                .header { background:linear-gradient(135deg,#111827 0%,#312e81 55%,#0891b2 100%); color:white; padding:32px 38px; }
                .header h1 { margin:0; font-size:38px; letter-spacing:-0.03em; }
                .header p { margin:10px 0 0; max-width:980px; color:#dbeafe; font-size:16px; line-height:1.5; }
                .container { max-width:1320px; margin:0 auto; padding:24px 28px 48px; }
                .card { background:white; border:1px solid #e5e7eb; border-radius:18px; box-shadow:0 10px 24px rgba(15,23,42,.06); padding:20px; }
                .control-card { margin-top:-30px; position:relative; z-index:2; }
                .kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin:20px 0; }
                .kpi-card { background:white; border:1px solid #e5e7eb; border-radius:18px; padding:18px; box-shadow:0 8px 20px rgba(15,23,42,.05); }
                .kpi-label { color:#6b7280; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
                .kpi-value { font-size:30px; font-weight:800; margin-top:8px; color:#111827; }
                .kpi-sub { color:#6b7280; font-size:13px; margin-top:4px; }
                .section-title { margin:28px 0 6px; font-size:26px; }
                .section-subtitle { color:#6b7280; margin-bottom:14px; line-height:1.5; }
                .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
                .grid-3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; }
                .pill { display:inline-block; padding:6px 10px; border-radius:999px; background:#eef2ff; color:#3730a3; font-size:13px; font-weight:700; margin-right:8px; }
                .note { color:#6b7280; font-size:13px; line-height:1.45; }
                .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table { font-size:13px; }
                label { font-weight:700; font-size:13px; color:#374151; }
                .prediction-box { border-radius:18px; border:1px solid #c7d2fe; background:#eef2ff; padding:18px; }
                .prediction-number { font-size:34px; font-weight:900; color:#312e81; }
                @media (max-width: 950px) { .kpi-grid,.grid-2,.grid-3 { grid-template-columns:1fr; } }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>{%config%}{%scripts%}{%renderer%}</footer>
        </body>
    </html>
    """

    app.layout = html.Div([
        html.Div([
            html.H1("Movie Success Predictor"),
            html.P("Explore 50k+ cleaned TMDB/Wikidata movie records, inspect revenue patterns, and test model predictions for revenue and blockbuster likelihood."),
        ], className="header"),

        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Genre filter"),
                        dcc.Dropdown(id="genre-filter", options=[{"label": g, "value": g} for g in genres], multi=True, placeholder="All genres"),
                    ]),
                    html.Div([
                        html.Label("Release year"),
                        dcc.RangeSlider(id="year-filter", min=year_min, max=year_max, value=[max(year_min, 1980), year_max],
                                        marks={year_min: str(year_min), 1980: "1980", 2000: "2000", year_max: str(year_max)}, step=1),
                    ]),
                    html.Div([
                        html.Label("Revenue range"),
                        dcc.RangeSlider(id="revenue-filter", min=0, max=rev_max, value=[0, rev_max], step=max(1, rev_max/100),
                                        marks={0: "$0", rev_max: money(rev_max)}),
                    ]),
                ], className="grid-3"),
                html.Div([
                    dcc.Checklist(id="blockbuster-only", options=[{"label": "Show blockbusters only", "value": "yes"}], value=[], inline=True),
                    html.Span("Blockbuster = top 20% revenue in the cleaned modeling dataset.", className="note", style={"marginLeft":"12px"}),
                ], style={"marginTop":"14px"}),
            ], className="card control-card"),

            html.Div(id="kpi-row", className="kpi-grid"),

            section_title("Overview", "High-level distributions and trends. These charts update with the filters above."),
            html.Div([
                html.Div(dcc.Graph(id="revenue-hist"), className="card"),
                html.Div(dcc.Graph(id="genre-bar"), className="card"),
            ], className="grid-2"),

            html.Div([
                html.Div(dcc.Graph(id="budget-revenue-scatter"), className="card"),
                html.Div(dcc.Graph(id="year-trend"), className="card"),
            ], className="grid-2", style={"marginTop":"18px"}),

            section_title("Model performance", "Final models are trained from the same feature groups used in the notebook. The dashboard summarizes the chosen model family and metrics."),
            html.Div([
                html.Div([
                    html.Span("Regression", className="pill"),
                    html.H3("Histogram Gradient Boosting"),
                    html.P("Chosen because it achieved the lowest RMSE and highest R² in the notebook, slightly outperforming Random Forest and clearly beating Ridge and the mean baseline.", className="note"),
                    html.Div(f"RMSE: {metrics['reg_rmse']:.3f}", className="prediction-number"),
                    html.Div(f"R²: {metrics['reg_r2']:.3f}", className="note"),
                ], className="card"),
                html.Div([
                    html.Span("Classification", className="pill"),
                    html.H3(metrics["clf_name"]),
                    html.P("The classifier is evaluated with F1 and ROC-AUC because blockbuster status is moderately imbalanced. The SMOTE version emphasizes recall for successful films.", className="note"),
                    html.Div(f"ROC-AUC: {metrics['clf_auc']:.3f}", className="prediction-number"),
                    html.Div(f"F1: {metrics['clf_f1']:.3f}", className="note"),
                ], className="card"),
            ], className="grid-2"),

            section_title("Prediction sandbox", "Change feature values to see how the trained models respond. This is for exploration, not a guarantee of actual revenue."),
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Budget"),
                        dcc.Slider(id="input-budget", min=0, max=300_000_000, value=50_000_000, step=1_000_000,
                                   marks={0:"$0", 50_000_000:"$50M", 150_000_000:"$150M", 300_000_000:"$300M"}),
                    ], style={"marginBottom":"24px"}),
                    html.Div([
                        html.Label("Runtime"),
                        dcc.Slider(id="input-runtime", min=60, max=220, value=110, step=5,
                                   marks={60:"60", 110:"110", 160:"160", 220:"220"}),
                    ], style={"marginBottom":"24px"}),
                    html.Div([
                        html.Label("Vote average"),
                        dcc.Slider(id="input-vote", min=0, max=10, value=6.8, step=0.1,
                                   marks={0:"0", 5:"5", 10:"10"}),
                    ], style={"marginBottom":"24px"}),
                    html.Div([
                        html.Label("Vote count"),
                        dcc.Slider(id="input-votecount", min=0, max=50000, value=1000, step=100,
                                   marks={0:"0", 1000:"1k", 10000:"10k", 50000:"50k"}),
                    ], style={"marginBottom":"24px"}),
                    html.Div([
                        html.Label("Popularity"),
                        dcc.Slider(id="input-popularity", min=0, max=500, value=30, step=1,
                                   marks={0:"0", 30:"30", 150:"150", 500:"500"}),
                    ]),
                    html.Div([
                        html.Div([html.Label("Genre"), dcc.Dropdown(id="input-genre", options=[{"label": g, "value": g} for g in genres], value=genres[0] if genres else "Unknown")]),
                        html.Div([html.Label("Language"), dcc.Input(id="input-language", type="text", value="en", style={"width":"100%"})]),
                        html.Div([html.Label("Release year"), dcc.Input(id="input-year", type="number", value=2024, style={"width":"100%"})]),
                    ], className="grid-3", style={"marginTop":"22px"}),
                ], className="card"),
                html.Div(id="prediction-output", className="prediction-box"),
            ], className="grid-2"),

            section_title("Movie table", "Filtered sample of the underlying cleaned dataset."),
            html.Div([
                dash_table.DataTable(
                    id="movie-table",
                    page_size=12,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "8px", "fontFamily":"Inter, sans-serif", "fontSize":"13px"},
                    style_header={"backgroundColor":"#eef2ff", "fontWeight":"bold", "color":"#312e81"},
                    style_data_conditional=[
                        {"if": {"filter_query": "{blockbuster} = 1"}, "backgroundColor": "#ecfeff"},
                    ],
                )
            ], className="card"),
        ], className="container"),
    ])

    @app.callback(
        Output("kpi-row", "children"),
        Output("revenue-hist", "figure"),
        Output("genre-bar", "figure"),
        Output("budget-revenue-scatter", "figure"),
        Output("year-trend", "figure"),
        Output("movie-table", "data"),
        Output("movie-table", "columns"),
        Input("genre-filter", "value"),
        Input("year-filter", "value"),
        Input("revenue-filter", "value"),
        Input("blockbuster-only", "value"),
    )
    def update_dashboard(selected_genres, year_range, revenue_range, blockbuster_only):
        filtered = apply_filters(df, selected_genres, year_range, revenue_range, "yes" in (blockbuster_only or []))
        if filtered.empty:
            empty_fig = px.scatter(title="No movies match the selected filters")
            return [kpi_card("Movies", "0")], empty_fig, empty_fig, empty_fig, empty_fig, [], []

        kpis = [
            kpi_card("Movies", f"{len(filtered):,}", f"{filtered['tmdb_id'].nunique() if 'tmdb_id' in filtered else len(filtered):,} unique records"),
            kpi_card("Median revenue", money(filtered["revenue"].median()), "Filtered sample"),
            kpi_card("Blockbuster share", f"{100*filtered['blockbuster'].mean():.1f}%", "Top-revenue label"),
            kpi_card("Median vote avg", f"{filtered['vote_average'].median():.2f}", "Audience rating proxy"),
        ]

        hist = px.histogram(filtered, x="log_revenue", nbins=45, title="Log revenue distribution", labels={"log_revenue":"log(1 + revenue)"}, color="blockbuster", color_discrete_sequence=[COLOR_ACCENT, COLOR_PRIMARY])
        hist.update_layout(margin=dict(l=10,r=10,t=45,b=10), height=390, bargap=0.04)

        genre_df = filtered.groupby("primary_genre", dropna=False).agg(movies=("title", "count"), median_revenue=("revenue", "median")).reset_index().sort_values("movies", ascending=False).head(15)
        genre_fig = px.bar(genre_df, x="movies", y="primary_genre", orientation="h", title="Top genres by movie count", hover_data={"median_revenue":":,.0f"}, color="median_revenue", color_continuous_scale="Blues")
        genre_fig.update_layout(margin=dict(l=10,r=10,t=45,b=10), height=390, yaxis={"categoryorder":"total ascending"})

        sample = filtered.sample(min(8000, len(filtered)), random_state=RANDOM_STATE)
        scatter = px.scatter(sample, x="log_budget", y="log_revenue", color="blockbuster", hover_name="title", title="Budget vs revenue", labels={"log_budget":"log(1 + budget)", "log_revenue":"log(1 + revenue)"}, color_discrete_sequence=[COLOR_ACCENT, COLOR_PRIMARY], opacity=0.55)
        scatter.update_layout(margin=dict(l=10,r=10,t=45,b=10), height=430)

        yearly = filtered.groupby("release_year").agg(movies=("title", "count"), median_revenue=("revenue", "median"), blockbuster_share=("blockbuster", "mean")).reset_index()
        trend = go.Figure()
        trend.add_trace(go.Scatter(x=yearly["release_year"], y=yearly["movies"], name="Movies", mode="lines", line=dict(color=COLOR_PRIMARY)))
        trend.add_trace(go.Scatter(x=yearly["release_year"], y=yearly["blockbuster_share"]*yearly["movies"].max(), name="Blockbuster share (scaled)", mode="lines", line=dict(color=COLOR_WARNING)))
        trend.update_layout(title="Movies over time and scaled blockbuster share", margin=dict(l=10,r=10,t=45,b=10), height=430, yaxis_title="Count / scaled share")

        table_cols = [c for c in DISPLAY_COLS if c in filtered.columns]
        table_df = filtered[table_cols].sort_values("revenue", ascending=False).head(500).copy()
        for c in ["budget", "revenue"]:
            if c in table_df:
                table_df[c] = table_df[c].map(lambda x: money(x))
        columns = [{"name": c.replace("_", " ").title(), "id": c} for c in table_df.columns]
        return kpis, hist, genre_fig, scatter, trend, table_df.to_dict("records"), columns

    @app.callback(
        Output("prediction-output", "children"),
        Input("input-budget", "value"),
        Input("input-runtime", "value"),
        Input("input-vote", "value"),
        Input("input-votecount", "value"),
        Input("input-popularity", "value"),
        Input("input-genre", "value"),
        Input("input-language", "value"),
        Input("input-year", "value"),
    )
    def update_prediction(budget, runtime, vote, vote_count, popularity, genre, language, year):
        # Median feature row keeps unspecified advanced features realistic.
        base = assets["modeling_df"][assets["features"]].copy()
        row = {}
        for col in assets["features"]:
            if col in assets["numeric_features"]:
                row[col] = float(pd.to_numeric(base[col], errors="coerce").median())
            else:
                mode = base[col].mode(dropna=True)
                row[col] = mode.iloc[0] if len(mode) else "Unknown"

        row.update({
            "log_budget": np.log1p(max(float(budget or 0), 0)),
            "budget_missing": int((budget or 0) <= 0),
            "runtime": float(runtime or 0),
            "log_popularity": np.log1p(max(float(popularity or 0), 0)),
            "vote_average": float(vote or 0),
            "log_vote_count": np.log1p(max(float(vote_count or 0), 0)),
            "release_year": float(year or 2024),
            "release_month": 7,
            "primary_genre": genre or "Unknown",
            "original_language": language or "en",
            "release_decade": str((int(year or 2024)//10)*10),
        })
        X_new = pd.DataFrame([row])[assets["features"]]
        pred_log_rev = float(assets["reg_model"].predict(X_new)[0])
        pred_rev = max(np.expm1(pred_log_rev), 0)
        prob = float(assets["clf_model"].predict_proba(X_new)[0, 1])

        verdict = "Likely blockbuster" if prob >= 0.5 else "Less likely blockbuster"
        color = COLOR_SUCCESS if prob >= 0.5 else COLOR_WARNING
        return html.Div([
            html.Div("Predicted revenue", className="kpi-label"),
            html.Div(money(pred_rev), className="prediction-number"),
            html.Div(f"Predicted log revenue: {pred_log_rev:.2f}", className="note"),
            html.Hr(style={"border":"none", "borderTop":"1px solid #c7d2fe"}),
            html.Div("Blockbuster probability", className="kpi-label"),
            html.Div(f"{100*prob:.1f}%", className="prediction-number", style={"color": color}),
            html.Div(verdict, style={"fontWeight":"800", "color": color, "marginTop":"4px"}),
            html.P("Note: popularity, vote average, and vote count may be post-release signals, so this sandbox is best interpreted as an exploratory model, not a pre-release forecasting tool.", className="note"),
        ])

    return app


if __name__ == "__main__":
    app = make_app()
    app.run(debug=True, port=8050, dev_tools_ui=False, dev_tools_props_check=False)
