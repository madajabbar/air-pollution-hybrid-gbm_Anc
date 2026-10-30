#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full run: LightGBM/XGBoost + BiLSTM, 3 polutan x 3 horizon, epoch penuh.
Cache GBM per polutan (fit sekali), BiLSTM per (polutan, horizon)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid as H
import numpy as np, pandas as pd
from scipy.stats import boxcox
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

FULL_EPOCHS = 60
COMBO = [("lightgbm", "bilstm"), ("xgboost", "bilstm")]

# cache GBM fit per polutan -> (f_gbm, resid_train, resid_test, test)
gbm_cache = {}

def get_gbm(stage1, pollutant):
    key = (stage1, pollutant)
    if key in gbm_cache:
        return gbm_cache[key]
    df = H.load()
    s = df[pollutant]
    s_bc, lam = boxcox(s.clip(lower=1e-3))
    train = s_bc[:-H.TEST_SIZE_DAYS]; test = s_bc[-H.TEST_SIZE_DAYS:]
    X = H.make_features(pd.Series(s_bc, index=s.index))
    Xtr, Xte = X.iloc[:-H.TEST_SIZE_DAYS], X.iloc[-H.TEST_SIZE_DAYS:]
    f_gbm = H.fit_gbm(stage1, Xtr, train, Xte)
    resid_train = np.asarray(train)[-len(f_gbm):] - f_gbm
    resid_test = np.asarray(test)[-len(f_gbm):] - f_gbm
    gbm_cache[key] = (f_gbm, resid_train, resid_test, test)
    return gbm_cache[key]

def run_one(stage1, pollutant, horizon, epochs):
    f_gbm, resid_train, resid_test, test = get_gbm(stage1, pollutant)
    scaler = MinMaxScaler()
    scaler.fit(resid_train.reshape(-1, 1))
    rt_s = scaler.transform(resid_train.reshape(-1, 1)).flatten()
    re_s = scaler.transform(resid_test.reshape(-1, 1)).flatten()
    pred_res_s, yte = H.stage2_bilstm(rt_s, re_s, horizon, epochs)
    if pred_res_s is None:
        return None
    pred_res = scaler.inverse_transform(pred_res_s.reshape(-1, 1)).flatten()
    n = min(len(pred_res), len(resid_test))
    final = f_gbm[:n] + pred_res[:n]
    actual = np.asarray(test)[:n]
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    return {"stage1": stage1, "stage2": "bilstm", "horizon": horizon,
            "pollutant": pollutant,
            "rmse": np.sqrt(mean_squared_error(actual, final)),
            "mae": mean_absolute_error(actual, final),
            "mape": np.mean(np.abs((actual - final) / actual)) * 100}

if __name__ == "__main__":
    for st1 in ["lightgbm", "xgboost"]:
        for pol in H.POLLUTANTS:
            for hz in H.HORIZONS:
                r = run_one(st1, pol, hz, FULL_EPOCHS)
                if r:
                    print(f"[OK] {st1}+bilstm {pol} h{hz}: "
                          f"RMSE={r['rmse']:.4f} MAE={r['mae']:.4f} MAPE={r['mape']:.2f}%",
                          flush=True)
                else:
                    print(f"[SKIP] {st1}+bilstm {pol} h{hz}", flush=True)
    print("SELESAI FULL RUN", flush=True)
