from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings, requests
from datetime import datetime

warnings.filterwarnings("ignore")

app = FastAPI(
    title="ML Volatility Prediction",
    description="Random Forest + Neural Network — volatility prediction",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

STORE  = {}
STATUS = {"status": "idle", "message": ""}
FEATURES = ["lag1", "lag2", "lag5", "rolling_vol_5", "rolling_vol_22"]


# ── Schemas ───────────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    ticker:     str   = Field("SPY")
    start_date: str   = Field("2010-01-01")
    end_date:   str   = Field("2024-01-01")
    test_size:  float = Field(0.2, ge=0.1, le=0.4)

class CsvTrainRequest(BaseModel):
    csv_data:  str   = Field(..., description="CSV text with Date,Close columns")
    test_size: float = Field(0.2, ge=0.1, le=0.4)

class PredictRequest(BaseModel):
    lag1:           float = Field(..., description="Yesterday return %")
    lag2:           float = Field(..., description="2-day ago return %")
    lag5:           float = Field(..., description="5-day ago return %")
    rolling_vol_5:  float = Field(..., description="5-day rolling std")
    rolling_vol_22: float = Field(..., description="22-day rolling std")
    model: Literal["rf", "nn", "both"] = Field("both")


# ── Data helpers ──────────────────────────────────────────────────────────────
def fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    # Method 1: yfinance with browser headers
    try:
        import yfinance as yf
        session = requests.Session()
        session.headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        )
        t = yf.Ticker(ticker, session=session)
        hist = t.history(start=start, end=end, auto_adjust=True)
        if not hist.empty:
            return hist["Close"].dropna()
    except Exception:
        pass

    # Method 2: plain yfinance download
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True, timeout=20)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df["Close"].dropna()
    except Exception:
        pass

    raise ValueError(
        f"Yahoo Finance blocked data for '{ticker}' inside Docker. "
        "Use POST /train/csv and paste your price data instead."
    )


def build_features(prices: pd.Series) -> pd.DataFrame:
    ret = 100 * np.log(prices / prices.shift(1))
    df = pd.DataFrame({"ret": ret})
    df["lag1"]          = df["ret"].shift(1)
    df["lag2"]          = df["ret"].shift(2)
    df["lag5"]          = df["ret"].shift(5)
    df["rolling_vol_5"] = df["ret"].rolling(5).std()
    df["rolling_vol_22"]= df["ret"].rolling(22).std()
    df["target"]        = df["ret"].abs().shift(-1)
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def train_models(df: pd.DataFrame, test_size: float, ticker: str):
    global STORE, STATUS
    STATUS["message"] = f"Training on {len(df)} rows..."
    X = df[FEATURES].values
    y = df["target"].values
    split = int(len(X) * (1 - test_size))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte  = y[:split], y[split:]
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                                min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(Xtr_s, ytr)
    rf_r2   = round(float(r2_score(yte, rf.predict(Xte_s))), 4)
    rf_rmse = round(float(np.sqrt(mean_squared_error(yte, rf.predict(Xte_s)))), 6)

    nn = MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation="relu",
                       solver="adam", alpha=0.001, learning_rate="adaptive",
                       max_iter=500, early_stopping=True, random_state=42)
    nn.fit(Xtr_s, ytr)
    nn_r2   = round(float(r2_score(yte, nn.predict(Xte_s))), 4)
    nn_rmse = round(float(np.sqrt(mean_squared_error(yte, nn.predict(Xte_s)))), 6)

    fi     = {k: round(v,4) for k,v in zip(FEATURES, rf.feature_importances_)}
    winner = "Random Forest" if rf_r2 > nn_r2 else "Neural Network"

    STORE = {
        "rf": rf, "nn": nn, "scaler": sc, "ticker": ticker,
        "trained_at": datetime.utcnow().isoformat(),
        "n_train": split, "n_test": len(Xte),
        "metrics": {
            "random_forest":  {"r2": rf_r2,  "rmse": rf_rmse},
            "neural_network": {"r2": nn_r2,  "rmse": nn_rmse},
            "best_model": winner,
        },
        "feature_importance": fi,
    }
    STATUS = {"status": "done",
              "message": f"Done. Best: {winner}  RF R²={rf_r2}  NN R²={nn_r2}"}


def run_train_bg(req: TrainRequest):
    global STATUS
    STATUS = {"status": "running", "message": f"Fetching {req.ticker}..."}
    try:
        prices = fetch_prices(req.ticker, req.start_date, req.end_date)
        df     = build_features(prices)
        train_models(df, req.test_size, req.ticker)
    except Exception as e:
        STATUS = {"status": "error", "message": str(e)}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "service": "ML Volatility API v2",
        "docs": "/docs",
        "tip": "If /train fails use /train/csv with your own price data"
    }

@app.get("/health", tags=["Info"])
def health():
    return {"status": "ok", "trained": bool(STORE)}

@app.post("/train", tags=["Training"])
def train(req: TrainRequest, bg: BackgroundTasks):
    """Train using Yahoo Finance data (may be blocked inside Docker)."""
    if STATUS.get("status") == "running":
        raise HTTPException(409, "Already training")
    bg.add_task(run_train_bg, req)
    return {"message": "Training started", "poll": "/status"}

@app.post("/train/csv", tags=["Training"])
def train_csv(req: CsvTrainRequest):
    """
    Train using your own CSV data — no internet required.

    Paste CSV with Date,Close columns:

        Date,Close
        2010-01-04,113.33
        2010-01-05,113.63
        2010-01-06,113.71
        ...

    You can export this from Excel or any finance site.
    """
    global STATUS
    if STATUS.get("status") == "running":
        raise HTTPException(409, "Already training")
    STATUS = {"status": "running", "message": "Parsing CSV..."}
    try:
        from io import StringIO
        df_raw = pd.read_csv(StringIO(req.csv_data), parse_dates=["Date"])
        df_raw = df_raw.sort_values("Date").set_index("Date")
        prices = df_raw["Close"].dropna()
        if len(prices) < 50:
            STATUS = {"status": "idle", "message": ""}
            raise HTTPException(422, "Need at least 50 rows")
        df = build_features(prices)
        train_models(df, req.test_size, "csv_upload")
        return {"message": STATUS["message"], "rows_used": len(df)}
    except HTTPException:
        raise
    except Exception as e:
        STATUS = {"status": "error", "message": str(e)}
        raise HTTPException(500, str(e))

@app.get("/status", tags=["Training"])
def status():
    """Check training progress."""
    return STATUS

@app.get("/metrics", tags=["Results"])
def metrics():
    """R², RMSE, feature importance, best model."""
    if not STORE:
        raise HTTPException(404, "Run POST /train or /train/csv first")
    return {
        "ticker":             STORE["ticker"],
        "trained_at":         STORE["trained_at"],
        "train_samples":      STORE["n_train"],
        "test_samples":       STORE["n_test"],
        "metrics":            STORE["metrics"],
        "feature_importance": STORE["feature_importance"],
    }

@app.post("/predict", tags=["Prediction"])
def predict(req: PredictRequest):
    """Predict next-day volatility from manual feature values."""
    if not STORE:
        raise HTTPException(404, "Train first")
    X  = np.array([[req.lag1, req.lag2, req.lag5,
                    req.rolling_vol_5, req.rolling_vol_22]])
    Xs = STORE["scaler"].transform(X)
    result = {}
    if req.model in ("rf", "both"):
        result["random_forest"] = round(float(STORE["rf"].predict(Xs)[0]), 6)
    if req.model in ("nn", "both"):
        result["neural_network"] = round(float(STORE["nn"].predict(Xs)[0]), 6)
    if req.model == "both":
        result["ensemble"] = round(float(np.mean(list(result.values()))), 6)
    return {"predictions": result}

@app.post("/predict/live", tags=["Prediction"])
def predict_live(ticker: str = Query("SPY")):
    """Fetch latest data from Yahoo Finance and predict."""
    if not STORE:
        raise HTTPException(404, "Train first")
    try:
        prices = fetch_prices(
            ticker,
            (pd.Timestamp.today()-pd.DateOffset(months=3)).strftime("%Y-%m-%d"),
            pd.Timestamp.today().strftime("%Y-%m-%d"),
        )
        df  = build_features(prices)
        row = df.iloc[-1]
        X   = row[FEATURES].values.reshape(1, -1)
        Xs  = STORE["scaler"].transform(X)
        rf_pred = round(float(STORE["rf"].predict(Xs)[0]), 6)
        nn_pred = round(float(STORE["nn"].predict(Xs)[0]), 6)
        return {
            "ticker":  ticker,
            "as_of":   str(df.index[-1].date()),
            "features":{f: round(float(v),4) for f,v in zip(FEATURES,X[0])},
            "predictions": {
                "random_forest":  rf_pred,
                "neural_network": nn_pred,
                "ensemble":       round((rf_pred+nn_pred)/2, 6),
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))
