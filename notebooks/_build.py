import os, json
# Spesifikasi 5 notebook eksperimen
specs = {
"01_stage1_SARIMA_vs_GBM": {
  "title": "Eksperimen 1: Tahap-1 SARIMA vs GBM (Fair Compare 45 Kombinasi)",
  "story": [
    ("md","# Eksperimen 1: Tahap-1 SARIMA vs GBM\n\nMembandingkan **SARIMA** (rujukan [7]) melawan **LightGBM** dan **XGBoost** sebagai tahap-1, dengan tahap-2 neural **BiLSTM/Attention/Fourier** yang identik. Protokol apple-to-apple: test 720 hari, window 120 hari, horizon 1/7/30."),
    ("code","import sys\nBASE='/home/mada/Documents/Kuliah/Algorithms and Complexity/02_Kode/hybrid-lgbm-xgb'\nsys.path.insert(0,BASE)\nimport hybrid as H\nimport numpy as np, pandas as pd\nfrom scipy.stats import boxcox\nfrom sklearn.preprocessing import MinMaxScaler\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error\nprint('Data dimuat:', H.load().shape)"),
    ("code","import sys\nBASE='/home/mada/Documents/Kuliah/Algorithms and Complexity/02_Kode/hybrid-lgbm-xgb'\nsys.path.insert(0,BASE)\nimport pandas as pd, re\n# Muat hasil fair compare 45 (hash re-run penuh — sangat lambat)\nrows=[]\nfor l in open(f'{BASE}/hasil_ref_vs_gbm.txt'):\n    m=re.match(r'\\[OK\\] (\\S+)\\+([\\S+]+) (PM\\d+|NO2) h(\\d+): RMSE=([\\d.]+)', l)\n    if m:\n        rows.append(dict(stage1=m.group(1),stage2=m.group(2),\n                         pollutant=m.group(3),horizon=int(m.group(4)),\n                         rmse=float(m.group(5))))\ndf=pd.DataFrame(rows)\nprint(f'Total hasil dimuat: {len(df)}')\nprint(df.groupby(['stage1','stage2']).size())"),
    ("code","import pandas as pd\npiv=df.pivot_table(index=['stage1','stage2'],columns=['pollutant','horizon'],values='rmse')\npiv.style.highlight_min(axis=0)"),
    ("code","# Visualisasi: bar chart per polutan untuk BiLSTM\nimport matplotlib.pyplot as plt\npol='NO2'\nsel=df[(df.stage2=='bilstm')&(df.pollutant==pol)]\nfig,ax=plt.subplots(figsize=(9,4))\nfor s1,c in [('sarima','red'),('lightgbm','orange'),('xgboost','green')]:\n    d=sel[sel.stage1==s1].set_index('horizon').rmse.sort_index()\n    ax.plot(d.index.astype(str),d.values,marker='o',label=s1,color=c)\nax.set_title(f'RMSE per horizon — {pol} (BiLSTM stage-2)'); ax.set_ylabel('RMSE'); ax.legend(); ax.grid(alpha=0.3)\nplt.tight_layout(); plt.show()"),
  ],
},
"02_stage1_full_run_GBM_BiLSTM": {
  "title": "Eksperimen 2: Full Run Tahap-1 GBM + BiLSTM (18 Kombinasi)",
  "story": [
    ("md","# Eksperimen 2: Full Run GBM + BiLSTM\n\nMenjalankan **LightGBM** dan **XGBoost** (tahap-1) dengan **BiLSTM** (tahap-2) pada semua 3 polutan × 3 horizon = 18 kombinasi."),
    ("code","import pandas as pd, sys\nBASE='/home/mada/Documents/Kuliah/Algorithms and Complexity/02_Kode/hybrid-lgbm-xgb'\nsys.path.insert(0,BASE)\nlines=[l for l in open(f'{BASE}/hasil_full_bilstm.txt') if '[OK]' in l]\nimport re\ndata=[]\nfor l in lines:\n    m=re.match(r'\\[OK\\] (\\S+)\\+bilstm (\\S+) h(\\d+): RMSE=([\\d.]+) MAE=([\\d.]+) MAPE=([\\d.]+)%',l)\n    if m: data.append(dict(stage1=m.group(1),pol=m.group(2),hz=int(m.group(3)),\n                          rmse=float(m.group(4)),mae=float(m.group(5)),mape=float(m.group(6))))\ndf=pd.DataFrame(data)\nprint(df.pivot_table(index='stage1',columns=['pol','hz'],values='rmse'))"),
    ("code","import matplotlib.pyplot as plt\nfig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)\nfor ax,p in zip(axes,['PM25','PM10','NO2']):\n    d=df[df.pol==p]\n    for s1,c in [('lightgbm','orange'),('xgboost','green')]:\n        sub=d[d.stage1==s1].set_index('hz').rmse.sort_index()\n        ax.plot(sub.index.astype(str),sub.values,marker='o',label=s1,color=c)\n    ax.set_title(p); ax.grid(alpha=0.3)\naxes[0].set_ylabel('RMSE'); axes[2].legend(); fig.suptitle('Full run GBM+BiLSTM')\nplt.tight_layout(); plt.show()"),
  ],
},
"03_stage2_BiLSTM_vs_Theta_vs_TFT": {
  "title": "Eksperimen 3: Tahap-2 BiLSTM vs Theta vs TFT",
  "story": [
    ("md","# Eksperimen 3: Tahap-2 (Koreksi Residual)\n\nDengan **LightGBM** tahap-1, membandingkan **BiLSTM**, **Theta**, dan **TFT** sebagai tahap-2 (koreksi residual)."),
    ("code","import pandas as pd, re, sys\nBASE='/home/mada/Documents/Kuliah/Algorithms and Complexity/02_Kode/hybrid-lgbm-xgb'\nsys.path.insert(0,BASE)\nlines=[l for l in open(f'{BASE}/hasil_fair_full.txt') if '[OK]' in l]\ndata=[]\nfor l in lines:\n    m=re.match(r'\\[OK\\] (\\S+) h(\\d+) \\+(\\S+): RMSE=([\\d.]+)',l)\n    if m: data.append(dict(pol=m.group(1),hz=int(m.group(2)),s2=m.group(3),rmse=float(m.group(4))))\ndf=pd.DataFrame(data)\nprint(df.pivot_table(index='s2',columns=['pol','hz'],values='rmse'))"),
    ("code","import matplotlib.pyplot as plt\nfig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)\nfor ax,p in zip(axes,['PM25','PM10','NO2']):\n    d=df[df.pol==p].pivot(index='s2',columns='hz',values='rmse')\n    d.plot(kind='bar',ax=ax,title=p,legend=False)\n    ax.grid(alpha=0.3)\naxes[0].set_ylabel('RMSE'); axes[2].legend(loc='best'); fig.suptitle('Tahap-2: BiLSTM vs Theta vs TFT')\nplt.tight_layout(); plt.show()"),
  ],
},
"04_xgboost_Theta_vs_BiLSTM": {
  "title": "Eksperimen 4: XGBoost + Theta vs XGBoost + BiLSTM",
  "story": [
    ("md","# Eksperimen 4: XGBoost + Theta vs XGBoost + BiLSTM\n\nDengan **XGBoost** tahap-1, membandingkan **Theta** (tanpa neural) melawan **BiLSTM** sebagai koreksi residual."),
    ("code","import sys\nBASE='/home/mada/Documents/Kuliah/Algorithms and Complexity/02_Kode/hybrid-lgbm-xgb'\nsys.path.insert(0,BASE)\nimport hybrid as H\nfrom compare_ref_vs_gbm import evaluate\nimport numpy as np\n# jalankan XGBoost+Theta (9 sel)\ntheta={}\nfor p in ['PM25','PM10','NO2']:\n    for h in [1,7,30]:\n        r=evaluate('xgboost','theta',h,p)\n        if r: theta[(p,h)]=r['rmse']\nimport pandas as pd, re\nbilstm={}\nfor l in open(f'{BASE}/hasil_ref_vs_gbm.txt'):\n    m=re.match(r'\\[OK\\] xgboost\\+bilstm (\\S+) h(\\d+): RMSE=([\\d.]+)',l)\n    if m: bilstm[(m.group(1),int(m.group(2)))]=float(m.group(3))\nrows=[]\nfor p in ['PM25','PM10','NO2']:\n    for h in [1,7,30]:\n        rows.append(dict(pol=p,hz=h,BiLSTM=bilstm[(p,h)],Theta=theta[(p,h)]))\nprint(pd.DataFrame(rows).set_index(['pol','hz']))"),
    ("code","import matplotlib.pyplot as plt\nsub=pd.DataFrame(rows)\nfig,ax=plt.subplots(figsize=(9,4))\nx=np.arange(9); w=0.35\nax.bar(x-w,sub.BiLSTM,w,label='BiLSTM',color='green')\nax.bar(x+w,sub.Theta,w,label='Theta',color='red')\nax.set_xticks(x); ax.set_xticklabels([f'{r.pol} h{r.hz}' for _,r in sub.iterrows()],rotation=45,fontsize=8)\nax.set_ylabel('RMSE'); ax.set_title('XGBoost: BiLSTM vs Theta'); ax.legend(); ax.grid(alpha=0.3,axis='y')\nplt.tight_layout(); plt.show()"),
  ],
},
"05_final_model_XGBoost_BiLSTM": {
  "title": "Eksperimen 5: Model Final XGBoost + BiLSTM",
  "story": [
    ("md","# Eksperimen 5: Model Final XGBoost + BiLSTM\n\nModel terpilih untuk paper: **XGBoost + BiLSTM**. Plot fit pada NO2 h1."),
    ("code","import sys\nBASE='/home/mada/Documents/Kuliah/Algorithms and Complexity/02_Kode/hybrid-lgbm-xgb'\nsys.path.insert(0,BASE)\nimport hybrid as H\nimport numpy as np, pandas as pd\nfrom scipy.stats import boxcox\nfrom sklearn.preprocessing import MinMaxScaler\nfrom sklearn.metrics import mean_squared_error\n\nwindow=120; test_len=720; poll='NO2'; hz=1\ndf=H.load()\ns_bc,_=boxcox(df[poll].clip(lower=1e-3))\ntrain,test=s_bc[:-test_len],s_bc[-test_len:]\nX=H.make_features(pd.Series(s_bc,index=df.index))\nf=H.fit_gbm('xgboost',X.iloc[:-test_len],train,X.iloc[-test_len:])\nresid_train=np.asarray(train)[-len(f):]-f\nresid_test=np.asarray(test)[-len(f):]-f\nfc_w=f[:window]; re_w=resid_test[:window]; test_w=test[:window]\nsc=MinMaxScaler(); sc.fit(resid_train.reshape(-1,1))\nrt=sc.transform(resid_train.reshape(-1,1)).flatten()\nrs=sc.transform(re_w.reshape(-1,1)).flatten()\npr,_=H.stage2_bilstm(rt,rs,hz,60)\npr=sc.inverse_transform(np.asarray(pr).reshape(-1,1)).flatten()\nn=min(len(pr),len(re_w))\nfinal=fc_w[:n]+pr[:n]; actual=test_w[:n]\nrmse=np.sqrt(mean_squared_error(actual,final))\nprint(f'RMSE NO2 h1 (window 120) = {rmse:.4f}')"),
    ("code","import matplotlib.pyplot as plt\nfig,ax=plt.subplots(figsize=(11,4))\nax.plot(actual,label='Aktual',color='black',lw=1.2)\nax.plot(final,label='XGBoost+BiLSTM',color='green',lw=1.4)\nax.set_title(f'Model Final: NO2 h1, RMSE={rmse:.3f}')\nax.set_xlabel('Hari test'); ax.set_ylabel('Box-Cox'); ax.legend(); ax.grid(alpha=0.3)\nplt.tight_layout(); plt.show()"),
  ],
},
}

import nbformat as nbf
base='/home/mada/Documents/Kuliah/Algorithms and Complexity/02_Kode/hybrid-lgbm-xgb/notebooks'
os.makedirs(base, exist_ok=True)
for name,spec in specs.items():
    nb=nbf.v4.new_notebook(); cells=[]
    for typ,content in spec['story']:
        cells.append(nbf.v4.new_markdown_cell(content) if typ=='md' else nbf.v4.new_code_cell(content))
    nb['cells']=cells
    nbf.write(nb, f"{base}/{name}.ipynb")
    print("dibuat:", name)
