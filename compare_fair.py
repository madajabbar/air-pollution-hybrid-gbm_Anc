#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perbandingan ADIL (apple-to-apple) tahap-2: BiLSTM vs Theta vs TFT.
Semua memprediksi TITIK TEST YANG SAMA, protokol identik.
- Stage-1: LightGBM (identik untuk semua)
- Test window: 120 hari terakhir (agar TFT walk-forward feasible)
- Horizon: 1
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
TEST_WINDOW = 120  # titik test SAMA untuk semua
STAGE2 = ["bilstm", "theta", "tft"]
EPOCHS = 60

def get_gbm(pollutant):
    df = H.load()
    s = df[pollutant]
    s_bc, lam = boxcox(s.clip(lower=1e-3))
    train = s_bc[:-H.TEST_SIZE_DAYS]; test = s_bc[-H.TEST_SIZE_DAYS:]
    X = H.make_features(pd.Series(s_bc, index=s.index))
    Xtr, Xte = X.iloc[:-H.TEST_SIZE_DAYS], X.iloc[-H.TEST_SIZE_DAYS:]
    f_gbm = H.fit_gbm("lightgbm", Xtr, train, Xte)
    resid_train = np.asarray(train)[-len(f_gbm):] - f_gbm
    resid_test = np.asarray(test)[-len(f_gbm):] - f_gbm
    return f_gbm, resid_train, resid_test, test

def evaluate(stage2, horizon, f_gbm, resid_train, resid_test, test):
    # ambil jendela test SAMA
    f_gbm_w = f_gbm[:TEST_WINDOW]
    resid_test_w = resid_test[:TEST_WINDOW]
    test_w = test[:TEST_WINDOW]
    if stage2 == 'theta':
        pred_res, _ = H.stage2_theta(resid_train, resid_test_w, horizon)
        pred_res = np.asarray(pred_res)[:TEST_WINDOW]
    else:
        scaler = MinMaxScaler()
        scaler.fit(resid_train.reshape(-1, 1))
        rt_s = scaler.transform(resid_train.reshape(-1, 1)).flatten()
        re_s = scaler.transform(resid_test_w.reshape(-1, 1)).flatten()
        if stage2 == 'bilstm':
            pred_res_s, _ = H.stage2_bilstm(rt_s, re_s, horizon, EPOCHS)
        else:
            pred_res_s, _ = H.stage2_tft(rt_s, re_s, horizon, EPOCHS, n_walk=TEST_WINDOW)
        pred_res = scaler.inverse_transform(pred_res_s.reshape(-1, 1)).flatten()
    n = min(len(pred_res), len(resid_test_w))
    final = f_gbm_w[:n] + pred_res[:n]
    actual = test_w[:n]
    return {"stage2": stage2, "n": n,
            "rmse": np.sqrt(mean_squared_error(actual, final)),
            "mae": mean_absolute_error(actual, final),
            "mape": np.mean(np.abs((actual - final) / actual)) * 100}

if __name__ == "__main__":
    gbm_cache = {}
    print(f"apple-to-apple: LightGBM + tahap-2, test_window={TEST_WINDOW} (identik)\n")
    for pol in POLLUTANTS:
        if pol not in gbm_cache:
            gbm_cache[pol] = get_gbm(pol)
        f_gbm, rt, rst, test = gbm_cache[pol]
        for hz in HORIZONS:
            for s2 in STAGE2:
                r = evaluate(s2, hz, f_gbm, rt, rst, test)
                print(f"[OK] {pol} h{hz} +{s2}: RMSE={r['rmse']:.4f} "
                      f"MAE={r['mae']:.4f} MAPE={r['mape']:.2f}% (n={r['n']})",
                      flush=True)
    print("\nSELESAI apple-to-apple")
