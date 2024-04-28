import pandas as pd
from sklearn.decomposition._pca import PCA
import numpy as np
from factor_analyzer import FactorAnalyzer
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def run():

    num_var = 3

    factors = pd.read_csv('factors.csv')

    if num_var == 7:
        variables = factors
    else:
        variables = factors[['VSQ(z)', 'Total Chi(z)', 'Aura(z)']]

    pca = PCA(n_components=3)
    fa = FactorAnalyzer(n_factors=3, rotation='varimax')
    num_bootstrap_sample = 1

    v_num = 2

    fa_results, pca_results, mean_results, loadings, components = [], [], [], [], []
    flat_avg_obs, pca_obs, efa_obs = [], [], []
    for i in tqdm(range(num_bootstrap_sample)):
        # get my bootstrapped sample
        sample = variables.sample(n=len(variables), replace=True)
        sample = sample - sample.mean()

        # firstly run PCA
        pca_data = pca.fit_transform(sample)
        pca_obs.append(pca_data[:, v_num])
        pca_results.append(np.var(pca_data, axis=0).tolist())
        components.append(pca.components_)
        
        # generata FA results
        fa.fit(sample)
        loadings.append(fa.loadings_)
        scores = fa.transform(sample)
        efa_obs.append(scores[:, v_num])
        fa_results.append(np.var(scores, axis=0).tolist())

        # generate mean results
        mean_results.append(sample.var().tolist())
        flat_avg_obs.append(sample.iloc[:, v_num])

if __name__=="__main__":
    run()