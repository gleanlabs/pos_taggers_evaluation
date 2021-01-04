import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
fig, ax = plt.subplots(figsize=(8,6))
Clrs = ['g','r']
for j,i in enumerate(['stanza' ,'spacy']):
    df = pd.read_csv('/Users/johanna/Desktop/{}_time_pandas_udfs.csv'.format(i))
    print(df)

    x = np.linspace(0,10)
    X = sm.add_constant(df['Unnamed: 0'].tolist())
    res = sm.OLS([i/3600 for i in df['sentences'].tolist()], X).fit()

    st, data, ss2 = summary_table(res, alpha=0.05)
    print(data)
    fittedvalues = data[:,2]
    predict_mean_se  = data[:,3]
    predict_ci_low, predict_ci_upp = data[:,6:8].T

    ax.plot(df['Unnamed: 0'].tolist(), [i/3600 for i in df['sentences'].tolist()], 'o', label="data_{}".format(i))
    ax.plot(df['Unnamed: 0'].tolist(), fittedvalues, color=Clrs[j], label='OLS_{}'.format(i))
    ax.plot(df['Unnamed: 0'].tolist(), predict_ci_low, 'b--')
    ax.plot(df['Unnamed: 0'].tolist(), predict_ci_upp, 'b--')

    new_data = sm.add_constant([1000000,5000000,10000000, 20000000, 50000000])
    result = res.get_prediction(new_data)
    predict_ci_low, predict_ci_upp = result.conf_int().T
    ax.plot([1000000,5000000,10000000, 20000000, 50000000], result.predicted_mean, 'o')
    ax.plot([1000000,5000000,10000000, 20000000, 50000000], result.predicted_mean, color=Clrs[j])
    ax.plot([1000000,5000000,10000000, 20000000, 50000000], predict_ci_low, 'b--')
    ax.plot([1000000,5000000,10000000, 20000000, 50000000], predict_ci_upp, 'b--')
color=Clrs[j],
ax.legend()
plt.xlabel('nb rows')
plt.ylabel('time in hours')
plt.show()
