#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perbandingan ADIL penuh: rujukan [7] vs GBM.
Apple-to-apple: protokol identik untuk semua.
- Tahap-1: SARIMA (rujukan) vs LightGBM vs XGBoost
- Tahap-2: BiLSTM / Attention / Fourier (semua rujukan)+
- test_window 120 hari, epoch 60, LightGBM stage1 identik
- 3 polutan x 3 horizon
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid as H
import numpy as np, pandas as pd
from scipy.stats import boxcox
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

POLLUTANTS = ["PM25", "PM10", "NO2"]
HORIZONS = [1, 7, 30]
TEST_WINDOW = 120
EPOCHS = 60

# kombinasi: (stage1, stage2)
COMBOS = [
    ("sarima", "bilstm"),   # rujukan [7]
    ("sarima", "attention"),
    ("sarima", "fourier"),
    ("lightgbm", "bilstm"), # usulan
    ("xgboost", "bilstm"),
]

# cache stage1 per polutan
stage1_cache = {}

def get_stage1(stage1, pollutant):
    key = (stage1, pollutant)
    if key in stage1_cache:
        return stage1_cache[key]
    df = H.load()
    s = df[pollutant]
    s_bc, lam = boxcox(s.clip(lower=1e-3))
    train = s_bc[:-H.TEST_SIZE_DAYS]; test = s_bc[-H.TEST_SIZE_DAYS:]
    if stage1 == 'sarima':
        fc, resid_train = H.fit_sarima(train, len(test), pollutant)
        # pecah resid_train: pakai seluruh data latih residual
        resid_train = np.asarray(resid_train)
        resid_test = np.asarray(test) - fc
        stage1_cache[key] = (fc, resid_train, resid_test, test)
        return stage1_cache[key]
    X = H.make_features(pd.Series(s_bc, index=s.index))
    Xtr, Xte = X.iloc[:-H.TEST_SIZE_DAYS], X.iloc[-H.TEST_SIZE_DAYS:]
    f_gbm = H.fit_gbm(stage1, Xtr, train, Xte)
    resid_train = np.asarray(train)[-len(f_gbm):] - f_gbm
    resid_test = np.asarray(test)[-len(f_gbm):] - f_gbm
    stage1_cache[key] = (f_gbm, resid_train, resid_test, test)
    return stage1_cache[key]

def stage2_dispatch(stage2, rt_s, re_s, horizon):
    if stage2 == 'bilstm':
        return H.stage2_bilstm(rt_s, re_s, horizon, EPOCHS)
    if stage2 == 'attention':
        return H.stage2_attention(rt_s, re_s, horizon, EPOCHS)
    if stage2 == 'fourier':
        return H.stage2_fourier(rt_s, re_s, horizon, EPOCHS)
    return None, None

def evaluate(stage1, stage2, horizon, pollutant):
    fc, resid_train, resid_test, test = get_stage1(stage1, pollutant)
    fc_w = fc[:TEST_WINDOW]
    resid_test_w = resid_test[:TEST_WINDOW]
    test_w = test[:TEST_WINDOW]
    # normalisasi residual untuk neural stage2
    scaler = MinMaxScaler()
    scaler.fit(resid_train.reshape(-1, 1))
    rt_s = scaler.transform(resid_train.reshape(-1, 1)).flatten()
    re_s = scaler.transform(resid_test_w.reshape(-1, 1)).flatten()
    pred_res_s, _ = stage2_dispatch(stage2, rt_s, re_s, horizon)
    if pred_res_s is None:
        return None
    pred_res = scaler.inverse_transform(np.asarray(pred_res_s).reshape(-1, 1)).flatten()
    n = min(len(pred_res), len(resid_test_w))
    final = fc_w[:n] + pred_res[:n]
    actual = test_w[:n]
    return {"stage1": stage1, "stage2": stage2, "horizon": horizon,
            "pollutant": pollutant, "n": n,
            "rmse": np.sqrt(mean_squared_error(actual, final)),
            "mae": mean_absolute_error(actual, final),
            "mape": np.mean(np.abs((actual - final) / actual)) * 100}

if __name__ == "__main__":
    results = []
    for pol in POLLUTANTS:
        for hz in HORIZONS:
            for s1, s2 in COMBOS:
                r = evaluate(s1, s2, hz, pol)
                if r:
                    results.append(r)
                    print(f"[OK] {s1}+{s2} {pol} h{hz}: RMSE={r['rmse']:.4f} "
                          f"MAE={r['mae']:.4f} MAPE={r['mape']:.2f}% (n={r['n']})",
                          flush=True)
                else:
                    print(f"[SKIP] {s1}+{s2} {pol} h{hz}", flush=True)
    print(f"\nSELESAI: {len(results)} hasil.")
