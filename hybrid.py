#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid 2-tahap: GBM (LightGBM/XGBoost) + koreksi residual neural (BiLSTM/TFT).
Analog paper rujukan [7] (SARIMA+BiLSTM), tapi SARIMA → GBM.

Kombinasi:
  1. lightgbm + bilstm
  2. xgboost + bilstm
  3. lightgbm + tft
  4. xgboost + tft

Run smoke (ringkas, non-full):  python hybrid.py --smoke
Run penuh (3 horizon x 3 polutan x 4 kombinasi):  python hybrid.py
"""
import os, sys, argparse, numpy as np, pandas as pd
from scipy.stats import boxcox
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Code4PollutantsBiLSTM", "transformed_data.csv")
TEST_SIZE_DAYS = 720
HORIZONS = [1, 7, 30]
POLLUTANTS = ["PM25", "PM10", "NO2"]
STAGE1 = ["lightgbm", "xgboost"]
STAGE2 = ["bilstm", "tft", "theta"]
SMOKE_EPOCHS = 12
FULL_EPOCHS = 60

# ---------- data ----------
def load():
    df = pd.read_csv(DATA)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')
    df = df.sort_values('Date').set_index('Date')
    return df

def make_features(s, lag=7):
    X = pd.DataFrame(index=s.index)
    for i in range(1, lag+1):
        X[f'lag{i}'] = s.shift(i)
    X['dow'] = s.index.dayofweek
    X['doy'] = s.index.dayofyear
    X['month'] = s.index.month
    return X

def fit_gbm(stage1, Xtr, ytr, Xte):
    ytr_s = pd.Series(np.asarray(ytr), index=Xtr.index) if not isinstance(ytr, pd.Series) else ytr
    mask = Xtr.notna().all(axis=1)
    Xf, yf = Xtr[mask], ytr_s[mask]
    if stage1 == 'lightgbm':
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                               num_leaves=31, verbose=-1)
    else:
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                              max_depth=5, verbosity=0)
    m.fit(Xf, yf)
    return m.predict(Xte)

def fit_sarima(y, test_len, pollutant=None):
    """SARIMA (rujukan [7]): fit pada train, forecast test_len langkah."""
    import warnings
    warnings.filterwarnings("ignore")
    from pmdarima import auto_arima
    order = {"PM25": (2,1,2), "PM10": (1,1,2), "NO2": (1,1,1)}.get(pollutant, (1,1,1))
    model = auto_arima(np.asarray(y), seasonal=True, m=12,
                       order=order, suppress_warnings=True,
                       error_action="ignore", stepwise=True, maxiter=40)
    fc = model.predict(n_periods=test_len)
    resid_train = np.asarray(model.resid())
    return np.asarray(fc), resid_train

# ---------- tahap 2: BiLSTM (keras) ----------
def stage2_bilstm(res_train, res_test, horizon, epochs):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    ts = max(1, horizon // 2)
    def seq(d):
        X, y = [], []
        for i in range(len(d) - ts):
            X.append(d[i:i+ts]); y.append(d[i+ts])
        return np.array(X), np.array(y)
    Xtr, ytr = seq(res_train); Xte, yte = seq(res_test)
    if len(Xtr) == 0 or len(Xte) == 0:
        return None, None
    Xtr = Xtr.reshape(len(Xtr), ts, 1); Xte = Xte.reshape(len(Xte), ts, 1)
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(ts, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(Xtr, ytr, epochs=epochs, batch_size=64, verbose=0)
    return model.predict(Xte, verbose=0).flatten(), yte

# ---------- tahap 2: TFT (pytorch-forecasting) ----------
def stage2_tft(res_train, res_test, horizon, epochs, n_walk=None):
    import torch
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    import lightning.pytorch as pl

    def frame(d, tag, base=0):
        idx = base + np.arange(len(d))
        return pd.DataFrame({"res": np.asarray(d), "time_idx": idx, "group": tag})

    if len(res_train) < 30 or len(res_test) < 5:
        return None, None

    # gabung train+test dgn time_idx global kontinu utk flurutan decoder
    enc_len = min(16, len(res_train) // 2)
    test_len = len(res_test)
    full = np.concatenate([np.asarray(res_train), np.asarray(res_test)])
    n = len(full)
    df = frame(full, "g", base=0)

    try:
        ds = TimeSeriesDataSet(
            df, time_idx="time_idx", target="res", group_ids=["group"],
            max_encoder_length=enc_len, max_prediction_length=horizon,
            static_categoricals=["group"], time_varying_known_categoricals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["res"],
            target_normalizer=GroupNormalizer(groups=["group"]),
        )
    except Exception:
        return None, None

    dl = ds.to_dataloader(train=True, batch_size=32, num_workers=0)
    tft = TemporalFusionTransformer.from_dataset(
        ds, learning_rate=1e-3, hidden_size=16, attention_head_size=2,
        dropout=0.1, hidden_continuous_size=8,
        loss=QuantileLoss([0.1, 0.5, 0.9]), log_interval=10, reduce_on_plateau_patience=2,
    )
    trainer = pl.Trainer(max_epochs=epochs, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, accelerator="cpu", limit_train_batches=30,
                         limit_val_batches=10)
    trainer.fit(tft, dl)

    # walk-forward: prediksi horizon langkah dari tiap titik test, maju per langkah
    if n_walk is None:
        n_walk = test_len
    preds = []
    # mulai dari encoder terakhir yg berakhir tepat sebelum test dimulai
    for i in range(0, min(n_walk, test_len), horizon):
        stop = len(res_train) + i
        if stop < enc_len:
            continue
        # potong data sampai stop, prediksi horizon ke depan
        sub = df.iloc[:stop].copy()
        try:
            ds_sub = TimeSeriesDataSet(
                sub, time_idx="time_idx", target="res", group_ids=["group"],
                max_encoder_length=enc_len, max_prediction_length=horizon,
                static_categoricals=["group"], time_varying_known_categoricals=[],
                time_varying_known_reals=["time_idx"], time_varying_unknown_reals=["res"],
                target_normalizer=GroupNormalizer(groups=["group"]),
            )
            dl_sub = ds_sub.to_dataloader(train=False, batch_size=32, num_workers=0)
            raw = tft.predict(dl_sub, mode="prediction", return_y=True)
            p = raw.output[0].flatten()
            preds.extend(p[:horizon])
        except Exception:
            continue
    if not preds:
        return None, None
    return np.asarray(preds)[:test_len], np.asarray(res_test)[:len(preds)]

# ---------- tahap 2: Theta (statsmodels) ----------
def stage2_theta(res_train, res_test, horizon):
    from statsmodels.tsa.forecasting.theta import ThetaModel
    try:
        model = ThetaModel(np.asarray(res_train), period=12, deseasonalize=True,
                           method="auto")
        fit = model.fit()
        fc = fit.forecast(steps=len(res_test))
        return np.asarray(fc), np.asarray(res_test)
    except Exception:
        return None, None

# ---------- tahap 2: Attention (BiLSTM + attention, rujukan) ----------
def stage2_attention(res_train, res_test, horizon, epochs):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Attention, Concatenate, Input, Permute, Multiply, Lambda
    try:
        ts = max(1, horizon // 2 if horizon > 1 else 4)
        n = len(res_train) - ts
        if n < 10: return None, None
        X, y = [], []
        for i in range(n):
            X.append(res_train[i:i+ts]); y.append(res_train[i+ts])
        X = np.asarray(X).reshape(-1, ts, 1); y = np.asarray(y)
        from sklearn.model_selection import train_test_split
        Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
        inp = Input(shape=(ts, 1))
        x = Bidirectional(LSTM(32, return_sequences=True))(inp)
        a = Attention()([x, x])
        from tensorflow.keras.layers import Flatten
        x = Flatten()(a)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        out = Dense(1)(x)
        from tensorflow.keras.models import Model
        model = Model(inp, out)
        model.compile(optimizer='adam', loss='mse')
        model.fit(Xtr, ytr, epochs=epochs, batch_size=64, validation_data=(Xv, yv),
                  verbose=0)
        # prediksi rekursif atas test
        preds = []
        state = list(res_train[-ts:])
        for _ in range(len(res_test)):
            px = np.asarray(state[-ts:]).reshape(1, ts, 1)
            p = float(model.predict(px, verbose=0)[0, 0])
            preds.append(p); state.append(p)
        return np.asarray(preds), np.asarray(res_test)
    except Exception:
        return None, None

# ---------- tahap 2: Fourier (R) + BiLSTM, rujukan) ----------
def stage2_fourier(res_train, res_test, horizon, epochs):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
    try:
        ts = max(1, horizon // 2 if horizon > 1 else 4)
        n = len(res_train) - ts
        if n < 10: return None, None
        # sinyal fourier per langkah (2 term: sin & cos, periode 12)
        def fourier_sig(k):
            t = np.arange(k)
            return np.stack([np.sin(2*np.pi*t/12), np.cos(2*np.pi*t/12)], axis=1)
        fw = fourier_sig(len(res_train))   # (len, 2)
        X, y = [], []
        for i in range(n):
            rt_seg = np.asarray(res_train[i:i+ts])          # (ts,)
            f_seg = fw[i:i+ts]                              # (ts, 2)
            X.append(np.stack([rt_seg, f_seg[:,0]], axis=1))  # (ts, 2)
            y.append(res_train[i+ts])
        X = np.asarray(X).reshape(-1, ts, 2); y = np.asarray(y)
        from sklearn.model_selection import train_test_split
        Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Sequential([
            Bidirectional(LSTM(32, return_sequences=True), input_shape=(ts, 2)),
            LSTM(32),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(Xtr, ytr, epochs=epochs, batch_size=64, validation_data=(Xv, yv),
                  verbose=0)
        fte = fourier_sig(len(res_test))
        preds = []
        state = list(res_train[-ts:])
        for i in range(len(res_test)):
            sf = np.asarray(state[-ts:])
            fxi = np.tile(fte[i], (ts, 1))
            px = np.stack([sf, fxi[:,0]], axis=1).reshape(1, ts, 2)
            p = float(model.predict(px, verbose=0)[0, 0])
            preds.append(p); state.append(p)
        return np.asarray(preds), np.asarray(res_test)
    except Exception as e:
        print(f"[WARN fourier] {e}")
        return None, None

# ---------- pipeline hybrid ----------
def run_hybrid(stage1, stage2, horizon, pollutant, epochs):
    df = load()
    s = df[pollutant]
    s_bc, lam = boxcox(s.clip(lower=1e-3))
    train = s_bc[:-TEST_SIZE_DAYS]; test = s_bc[-TEST_SIZE_DAYS:]

    X = make_features(pd.Series(s_bc, index=s.index))
    Xtr, Xte = X.iloc[:-TEST_SIZE_DAYS], X.iloc[-TEST_SIZE_DAYS:]
    ytr = train

    f_gbm = fit_gbm(stage1, Xtr, ytr, Xte)
    resid_train = np.asarray(train)[-len(f_gbm):] - f_gbm
    resid_test = np.asarray(test)[-len(f_gbm):] - f_gbm

    # Theta: model statistik, gunakan residual mentah (tanpa scaler)
    if stage2 == 'theta':
        pred_res, yte = stage2_theta(resid_train, resid_test, horizon)
        if pred_res is None:
            return None
        pred_res = np.asarray(pred_res)
    else:
        # neural: skala residual utk stabilisasi training
        scaler = MinMaxScaler()
        scaler.fit(resid_train.reshape(-1, 1))
        rt_s = scaler.transform(resid_train.reshape(-1, 1)).flatten()
        re_s = scaler.transform(resid_test.reshape(-1, 1)).flatten()
        if stage2 == 'bilstm':
            pred_res_s, yte = stage2_bilstm(rt_s, re_s, horizon, epochs)
        else:
            pred_res_s, yte = stage2_tft(rt_s, re_s, horizon, epochs, n_walk=20)
        if pred_res_s is None:
            return None
        pred_res = scaler.inverse_transform(pred_res_s.reshape(-1, 1)).flatten()

    n = min(len(pred_res), len(resid_test))
    final = f_gbm[:n] + pred_res[:n]
    actual = np.asarray(test)[:n]

    rmse = np.sqrt(mean_squared_error(actual, final))
    mae = mean_absolute_error(actual, final)
    mape = np.mean(np.abs((actual - final) / actual)) * 100
    return {"stage1": stage1, "stage2": stage2, "horizon": horizon,
            "pollutant": pollutant, "rmse": rmse, "mae": mae, "mape": mape}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    epochs = SMOKE_EPOCHS if a.smoke else FULL_EPOCHS
    # smoke: 1 kombinasi, horizon 1, PM25 saja
    combos = [("lightgbm", "bilstm"), ("xgboost", "bilstm"),
              ("lightgbm", "tft"), ("xgboost", "tft"),
              ("lightgbm", "theta"), ("xgboost", "theta")]
    if a.smoke:
        combos = [("lightgbm", "bilstm")]
        HORIZONS_S = [1]; POLLUTANTS_S = ["PM25"]
    else:
        HORIZONS_S = HORIZONS; POLLUTANTS_S = POLLUTANTS

    rows = []
    for st1, st2 in combos:
        for hz in HORIZONS_S:
            for pol in POLLUTANTS_S:
                r = run_hybrid(st1, st2, hz, pol, epochs)
                if r:
                    rows.append(r)
                    print(f"[OK] {st1}+{st2} {pol} h{hz}: "
                          f"RMSE={r['rmse']:.4f} MAE={r['mae']:.4f} MAPE={r['mape']:.2f}%")
                else:
                    print(f"[SKIP] {st1}+{st2} {pol} h{hz}")
    print("\nSELESAI —", len(rows), "hasil.")

if __name__ == "__main__":
    main()
