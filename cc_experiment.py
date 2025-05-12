"""
The experiment MAIN for Communities and Crime.
 * Run the file and the CC experiments will complete
 * See compas experiment file for more details on how to read results.
"""
import warnings
warnings.filterwarnings('ignore') 

from adversarial_models import * 
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from tqdm import tqdm


import lime
import lime.lime_tabular
import shap
import pandas as pd
from evals import get_pgi, get_pgi_multi, get_bex

from copy import deepcopy

import random
random.seed(42)

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
X, y, cols = get_and_preprocess_cc(params)

# add unrelated columns, setup
# X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
# X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])

X['unrelated_column_one'] = np.random.uniform(-1, 1, size=X.shape[0])
X['unrelated_column_two'] = np.random.uniform(-1, 1, size=X.shape[0])
features = [c for c in X]

sens_feature = 'racePctWhite numeric'
foil_feature1 = 'unrelated_column_one'
foil_feature2 = 'unrelated_column_two'

race_indc = features.index('racePctWhite numeric')
unrelated_indcs = features.index('unrelated_column_one')
unrelated_indcs1 = features.index('unrelated_column_two')

X = X.values

###
## The models f and psi for CC.  We discriminate based on race for f and concider two RANDOMLY DRAWN features to display in psi
#

# the biased model 
class racist_model_f:
    # Decision rule: classify negatively if race is black
    def predict(self,X):
        return np.array([params.negative_outcome if x[race_indc] > 0 else params.positive_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
    
# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to randomly drawn column 'unrelated column'
    def predict_proba(self, X): 
        # result = np.zeros((X.shape[0], 2))
        # d = X[:, unrelated_indcs].flatten()
        # result[:, 1] = (d + 1) / 2
        # result[:, 0] = 1 - result[:, 1]
        # result[result < 0] = 0
        # result[result > 1] = 1
        # return result
        return one_hot_encode(np.array([params.negative_outcome if x[unrelated_indcs] > 0 else params.positive_outcome for x in X]))

# the display model with two unrelated features
class innocuous_model_psi_two:
	def predict_proba(self, X):
    # def predict_proba(self, X):
    #     result = np.zeros((X.shape[0], 2))
    #     # d1 = X[:, unrelated_indcs].flatten()
    #     # d2 = X[:, unrelated_indcs].flatten()
    #     d = (X[:, unrelated_indcs].flatten() + X[:, unrelated_indcs1].flatten()) / 2
    #     result[:, 1] = (d + 1) / 2
    #     result[:, 0] = 1 - result[:, 1]
    #     result[result < 0] = 0
    #     result[result > 1] = 1
    #     return result
		# Using 0.5 to make it easier to detect decision boundary on perturbation
		A = np.where(X[:,unrelated_indcs] > 0, params.positive_outcome, params.negative_outcome)
		B = np.where(X[:,unrelated_indcs1] < 0, params.positive_outcome, params.negative_outcome)
		preds = np.logical_xor(A, B).astype(int)
		return one_hot_encode(preds)
#
##
###

def experiment_main(verbose=False):
	"""
	Run through experiments for LIME/SHAP on CC using both one and two unrelated features.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""
	np.random.seed(42)
	rfeature_1, rfeature_2, rfeature_3 = random.sample(features, 3)
	
	xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)
	ss = StandardScaler().fit(xtrain)
	xtrain = ss.transform(xtrain)
	xtest = ss.transform(xtest)

	xtest_df = pd.DataFrame(xtest, columns=features)

	if verbose:
		print('---------------------')
		print("Beginning LIME CC Experiments....")
		print("(These take some time to run because we have to generate explanations for every point in the test set) ")
		print('---------------------')

	# Train the adversarial model for LIME with f and psi 
	# model 1
	adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, categorical_features=[features.index('unrelated_column_one'), features.index('unrelated_column_two')], feature_names=features, perturbation_multiplier=30)
	if verbose:
		print(f'Model LIME1 trained with fidelity {adv_lime.fidelity(xtest)}')
	else:
		print(f'\t %cc LIME-1 fidelity {adv_lime.fidelity(xtest)}')
		
	pgi_results = []
	bex_results = []
	for feature in features:
		if feature not in [sens_feature, foil_feature1, foil_feature2]:
			pgi_score = get_pgi(xtest_df, adv_lime, feature)
			bex_score = get_bex(xtest_df, adv_lime, feature)
			pgi_results.append((feature, np.mean(pgi_score)))
			bex_results.append((feature, np.mean(bex_score)))
	sens_lime_pgi = get_pgi(xtest_df, adv_lime, sens_feature)
	foil_lime1_pgi = get_pgi(xtest_df, adv_lime, foil_feature1)
	foil_lime2_pgi = get_pgi(xtest_df, adv_lime, foil_feature2)
	
	sens_lime_bex = get_bex(xtest_df, adv_lime, sens_feature)
	foil_lime1_bex = get_bex(xtest_df, adv_lime, foil_feature1)
	foil_lime2_bex = get_bex(xtest_df, adv_lime, foil_feature2)
	
	pgu_results = []
	for feature in features:
		inv_features = [f for f in features if f != feature]
		if feature not in [sens_feature, foil_feature1, foil_feature2]:
			pgu_score = 0-get_pgi_multi(xtest_df, adv_lime, inv_features)
			pgu_results.append((feature, np.mean(pgu_score)))
	sens_lime_pgu = 0-get_pgi_multi(xtest_df, adv_lime, [f for f in features if f != sens_feature])
	foil_lime1_pgu = 0-get_pgi_multi(xtest_df, adv_lime, [f for f in features if f != foil_feature1])
	foil_lime2_pgu = 0-get_pgi_multi(xtest_df, adv_lime, [f for f in features if f != foil_feature2])

	if verbose:
		print(f'Mean PGI (random) for adv_lime model: {np.mean([score for _, score in pgi_results])}, Std Dev: {np.std([score for _, score in pgi_results])}')
		print(f'(L) SENS Mean {sens_feature} PGI for adv_lime model: {np.mean(sens_lime_pgi)}')
		print(f'(H) FOIL1 Mean {foil_feature1} PGI for adv_lime model: {np.mean(foil_lime1_pgi)}')
		print(f'(?) foil2 Mean {foil_feature2} PGI for adv_lime model: {np.mean(foil_lime2_pgi)}')
		print(f'Mean BEX (random) for adv_lime model: {np.mean([score for _, score in bex_results])}, Std Dev: {np.std([score for _, score in bex_results])}')
		print(f'(L) SENS Mean {sens_feature} BEX for adv_lime model: {np.mean(sens_lime_bex)}')
		print(f'(H) FOIL1 Mean {foil_feature1} BEX for adv_lime model: {np.mean(foil_lime1_bex)}')
		print(f'(?) foil2 Mean {foil_feature2} BEX for adv_lime model: {np.mean(foil_lime2_bex)}')
	else:
		print(f' cc1 & lime & PGI & {np.mean(sens_lime_pgi):.3f} & {np.mean(foil_lime1_pgi):.3f} & {np.mean(foil_lime2_pgi):.3f} & {np.mean([score for _, score in pgi_results]):.3f} \\\\ ')
		print(f' cc1 & lime & PGU & {np.mean(sens_lime_pgu):.3f} & {np.mean(foil_lime1_pgu):.3f} & {np.mean(foil_lime2_pgu):.3f} & {np.mean([score for _, score in pgu_results]):.3f} \\\\ ')
		print(f' cc1 & lime & AXE & {np.mean(sens_lime_bex):.3f} & {np.mean(foil_lime1_bex):.3f} & {np.mean(foil_lime2_bex):.3f} & {np.mean([score for _, score in bex_results]):.3f} \\\\ ')

	
	# model 2
	adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain, feature_names=features, perturbation_multiplier=30, categorical_features=[features.index('unrelated_column_one'), features.index('unrelated_column_two')])
	if verbose:
		print(f'Model LIME2 trained with fidelity {adv_lime.fidelity(xtest)}')
	else:
		print(f'\t %cc LIME-2 fidelity {adv_lime.fidelity(xtest)}')
		
	pgi_results = []
	bex_results = []
	for feature in features:
		if feature not in [sens_feature, foil_feature1, foil_feature2]:
			pgi_score = get_pgi(xtest_df, adv_lime, feature)
			bex_score = get_bex(xtest_df, adv_lime, feature)
			pgi_results.append((feature, np.mean(pgi_score)))
			bex_results.append((feature, np.mean(bex_score)))
	
	sens_lime_pgi = get_pgi(xtest_df, adv_lime, sens_feature)
	foil_lime1_pgi = get_pgi(xtest_df, adv_lime, foil_feature1)
	foil_lime2_pgi = get_pgi(xtest_df, adv_lime, foil_feature2)
	
	sens_lime_bex = get_bex(xtest_df, adv_lime, sens_feature)
	foil_lime1_bex = get_bex(xtest_df, adv_lime, foil_feature1)
	foil_lime2_bex = get_bex(xtest_df, adv_lime, foil_feature2)

	pgu_results = []
	for feature in features:
		inv_features = [f for f in features if f != feature]
		if feature not in [sens_feature, foil_feature1, foil_feature2]:
			pgu_score = 0-get_pgi_multi(xtest_df, adv_lime, inv_features)
			pgu_results.append((feature, np.mean(pgu_score)))
	sens_lime_pgu = 0-get_pgi_multi(xtest_df, adv_lime, [f for f in features if f != sens_feature])
	foil_lime1_pgu = 0-get_pgi_multi(xtest_df, adv_lime, [f for f in features if f != foil_feature1])
	foil_lime2_pgu = 0-get_pgi_multi(xtest_df, adv_lime, [f for f in features if f != foil_feature2])
	
	if verbose:
		print(f'Mean PGI (random) for adv_lime model: {np.mean([score for _, score in pgi_results])}, Std Dev: {np.std([score for _, score in pgi_results])}')
		print(f'(L) SENS Mean {sens_feature} PGI for adv_lime model: {np.mean(sens_lime_pgi)}')
		print(f'(H) FOIL1 Mean {foil_feature1} PGI for adv_lime model: {np.mean(foil_lime1_pgi)}')
		print(f'(?) foil2 Mean {foil_feature2} PGI for adv_lime model: {np.mean(foil_lime2_pgi)}')
		print(f'Mean BEX (random) for adv_lime model: {np.mean([score for _, score in bex_results])}, Std Dev: {np.std([score for _, score in bex_results])}')
		print(f'(L) SENS Mean {sens_feature} BEX for adv_lime model: {np.mean(sens_lime_bex)}')
		print(f'(H) FOIL1 Mean {foil_feature1} BEX for adv_lime model: {np.mean(foil_lime1_bex)}')
		print(f'(?) foil2 Mean {foil_feature2} BEX for adv_lime model: {np.mean(foil_lime2_bex)}')
	else:
		print(f' cc2 & lime & PGI & {np.mean(sens_lime_pgi):.3f} & {np.mean(foil_lime1_pgi):.3f} & {np.mean(foil_lime2_pgi):.3f} & {np.mean([score for _, score in pgi_results]):.3f} \\\\ ')
		print(f' cc2 & lime & PGU & {np.mean(sens_lime_pgu):.3f} & {np.mean(foil_lime1_pgu):.3f} & {np.mean(foil_lime2_pgu):.3f} & {np.mean([score for _, score in pgu_results]):.3f} \\\\ ')
		print(f' cc2 & lime & AXE & {np.mean(sens_lime_bex):.3f} & {np.mean(foil_lime1_bex):.3f} & {np.mean(foil_lime2_bex):.3f} & {np.mean([score for _, score in bex_results]):.3f} \\\\ ')
	
	# model 3
	background_distribution = shap.kmeans(xtrain, 10)
	adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features)
	if verbose:
		print(f'Model SHAP1 trained with fidelity {adv_shap.fidelity(xtest)}')
	else:
		print(f'\t %cc SHAP-1 fidelity {adv_shap.fidelity(xtest)}')
		
	pgi_results = []
	bex_results = []
	for feature in features:
		if feature not in [sens_feature, foil_feature1, foil_feature2]:
			pgi_score = get_pgi(xtest_df, adv_shap, feature)
			bex_score = get_bex(xtest_df, adv_shap, feature)
			pgi_results.append((feature, np.mean(pgi_score)))
			bex_results.append((feature, np.mean(bex_score)))
		
	sens_shap_pgi = get_pgi(xtest_df, adv_shap, sens_feature)
	foil_shap1_pgi = get_pgi(xtest_df, adv_shap, foil_feature1)
	foil_shap2_pgi = get_pgi(xtest_df, adv_shap, foil_feature2)
	
	sens_shap_bex = get_bex(xtest_df, adv_shap, sens_feature)
	foil_shap1_bex = get_bex(xtest_df, adv_shap, foil_feature1)
	foil_shap2_bex = get_bex(xtest_df, adv_shap, foil_feature2)

	pgu_results = []
	for feature in features:
		inv_features = [f for f in features if f != feature]
		if feature not in [sens_feature, foil_feature1, foil_feature2]:
			pgu_score = 0-get_pgi_multi(xtest_df, adv_shap, inv_features)
			pgu_results.append((feature, np.mean(pgu_score)))
	sens_shap_pgu = 0-get_pgi_multi(xtest_df, adv_shap, [f for f in features if f != sens_feature])
	foil_shap1_pgu = 0-get_pgi_multi(xtest_df, adv_shap, [f for f in features if f != foil_feature1])
	foil_shap2_pgu = 0-get_pgi_multi(xtest_df, adv_shap, [f for f in features if f != foil_feature2])
		

	if verbose:
		print(f'Mean PGI (random) for adv_shap model: {np.mean([score for _, score in pgi_results])}, Std Dev: {np.std([score for _, score in pgi_results])}')
		print(f'(L) SENS Mean {sens_feature} PGI for adv_shap model: {np.mean(sens_shap_pgi)}')
		print(f'(H) FOIL1 Mean {foil_feature1} PGI for adv_shap model: {np.mean(foil_shap1_pgi)}')
		print(f'(?) foil2 Mean {foil_feature2} PGI for adv_shap model: {np.mean(foil_shap2_pgi)}')
		print(f'Mean BEX (random) for adv_shap model: {np.mean([score for _, score in bex_results])}, Std Dev: {np.std([score for _, score in bex_results])}')
		print(f'(L) SENS Mean {sens_feature} BEX for adv_shap model: {np.mean(sens_shap_bex)}')
		print(f'(H) FOIL1 Mean {foil_feature1} BEX for adv_shap model: {np.mean(foil_shap1_bex)}')
		print(f'(?) foil2 Mean {foil_feature2} BEX for adv_shap model: {np.mean(foil_shap2_bex)}')
	else:
		print(f' cc1 & shap & PGI & {np.mean(sens_shap_pgi):.3f} & {np.mean(foil_shap1_pgi):.3f} & {np.mean(foil_shap2_pgi):.3f} & {np.mean([score for _, score in pgi_results]):.3f} \\\\ ')
		print(f' cc1 & shap & PGU & {np.mean(sens_shap_pgu):.3f} & {np.mean(foil_shap1_pgu):.3f} & {np.mean(foil_shap2_pgu):.3f} & {np.mean([score for _, score in pgu_results]):.3f} \\\\ ')
		print(f' cc1 & shap & AXE & {np.mean(sens_shap_bex):.3f} & {np.mean(foil_shap1_bex):.3f} & {np.mean(foil_shap2_bex):.3f} & {np.mean([score for _, score in bex_results]):.3f} \\\\ ')
	
	# model 4
	background_distribution = shap.kmeans(xtrain, 10)
	adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain, feature_names=features)
	if verbose:
		print(f'Model SHAP2 trained with fidelity {adv_shap.fidelity(xtest)}')
	else:
		print(f'\t %cc SHAP-2 fidelity {adv_shap.fidelity(xtest)}')
		
	pgi_results = []
	bex_results = []
	for feature in features:
		if feature not in [sens_feature, foil_feature1, foil_feature2]:
			pgi_score = get_pgi(xtest_df, adv_shap, feature)
			bex_score = get_bex(xtest_df, adv_shap, feature)
			pgi_results.append((feature, np.mean(pgi_score)))
			bex_results.append((feature, np.mean(bex_score)))
	
	
	sens_shap_pgi = get_pgi(xtest_df, adv_shap, sens_feature)
	foil_shap1_pgi = get_pgi(xtest_df, adv_shap, foil_feature1)
	foil_shap2_pgi = get_pgi(xtest_df, adv_shap, foil_feature2)
	
	sens_shap_bex = get_bex(xtest_df, adv_shap, sens_feature)
	foil_shap1_bex = get_bex(xtest_df, adv_shap, foil_feature1)
	foil_shap2_bex = get_bex(xtest_df, adv_shap, foil_feature2)

	pgu_results = []
	for feature in features:
		inv_features = [f for f in features if f != feature]
		if feature not in [sens_feature, foil_feature1, foil_feature2]:
			pgu_score = 0-get_pgi_multi(xtest_df, adv_shap, inv_features)
			pgu_results.append((feature, np.mean(pgu_score)))
	sens_shap_pgu = 0-get_pgi_multi(xtest_df, adv_shap, [f for f in features if f != sens_feature])
	foil_shap1_pgu = 0-get_pgi_multi(xtest_df, adv_shap, [f for f in features if f != foil_feature1])
	foil_shap2_pgu = 0-get_pgi_multi(xtest_df, adv_shap, [f for f in features if f != foil_feature2])
		
	
	if verbose:
		print(f'Mean PGI (random) for adv_shap model: {np.mean([score for _, score in pgi_results])}, Std Dev: {np.std([score for _, score in pgi_results])}')
		print(f'(L) SENS Mean {sens_feature} PGI for adv_shap model: {np.mean(sens_shap_pgi)}')
		print(f'(H) FOIL1 Mean {foil_feature1} PGI for adv_shap model: {np.mean(foil_shap1_pgi)}')
		print(f'(?) foil2 Mean {foil_feature2} PGI for adv_shap model: {np.mean(foil_shap2_pgi)}')
		print(f'Mean BEX (random) for adv_shap model: {np.mean([score for _, score in bex_results])}, Std Dev: {np.std([score for _, score in bex_results])}')
		print(f'(L) SENS Mean {sens_feature} BEX for adv_shap model: {np.mean(sens_shap_bex)}')
		print(f'(H) FOIL1 Mean {foil_feature1} BEX for adv_shap model: {np.mean(foil_shap1_bex)}')
		print(f'(?) foil2 Mean {foil_feature2} BEX for adv_shap model: {np.mean(foil_shap2_bex)}')
	else:
		print(f' cc2 & shap & PGI & {np.mean(sens_shap_pgi):.3f} & {np.mean(foil_shap1_pgi):.3f} & {np.mean(foil_shap2_pgi):.3f} & {np.mean([score for _, score in pgi_results]):.3f} \\\\ ')
		print(f' cc2 & shap & PGU & {np.mean(sens_shap_pgu):.3f} & {np.mean(foil_shap1_pgu):.3f} & {np.mean(foil_shap2_pgu):.3f} & {np.mean([score for _, score in pgu_results]):.3f} \\\\ ')
		print(f' cc2 & shap & AXE & {np.mean(sens_shap_bex):.3f} & {np.mean(foil_shap1_bex):.3f} & {np.mean(foil_shap2_bex):.3f} & {np.mean([score for _, score in bex_results]):.3f} \\\\ ')

	return
	# print(f'Model trained with fidelity {adv_lime.fidelity(xtest)}')
	
	# features_without_foil2 = [f for f in features if f != foil_feature2]
	# feature_samples = [random.sample(features_without_foil2, 1 + (len(features_without_foil2) // 2)) for _ in range(100)]
	
	# feature_samples_both = [sample for sample in feature_samples if sens_feature in sample and foil_feature1 in sample]
	# feature_samples_sens_only = [sample for sample in feature_samples if sens_feature in sample and foil_feature1 not in sample]
	# feature_samples_foil_only = [sample for sample in feature_samples if foil_feature1 in sample and sens_feature not in sample]
	# feature_samples_neither = [sample for sample in feature_samples if sens_feature not in sample and foil_feature1 not in sample]
	
	# pgi_results_both = []
	# pgi_results_sens_only = []
	# pgi_results_foil_only = []
	# pgi_results_neither = []

	# for feature_sample_set, result_list in zip(
	# 	[feature_samples_both, feature_samples_sens_only, feature_samples_foil_only, feature_samples_neither],
	# 	[pgi_results_both, pgi_results_sens_only, pgi_results_foil_only, pgi_results_neither]
	# ):
	# 	for feature_sample in tqdm(feature_sample_set, desc="Processing feature samples"):
	# 		pgi_values = get_pgi_multi(xtest_df, adv_lime, feature_sample)
	# 		result_list.append(np.mean(pgi_values))
	# print(f'PGI results from: both, \t Mean: {np.mean(pgi_results_both)}, num unique explanations: {len(pgi_results_both)}')
	# print(f'PGI results from: sens only, \t Mean: {np.mean(pgi_results_sens_only)}, num unique explanations: {len(pgi_results_sens_only)}')
	# print(f'PGI results from: foil only, \t Mean: {np.mean(pgi_results_foil_only)}, num unique explanations: {len(pgi_results_foil_only)}')
	# print(f'PGI results from: neither, \t Mean: {np.mean(pgi_results_neither)}, num unique explanations: {len(pgi_results_neither)}')


	# return
	# sens_lime1_pgi = get_pgi(xtest_df, adv_lime, sens_feature)
	# foil_lime1_pgi = get_pgi(xtest_df, adv_lime, foil_feature1)

	# rfeature_1_pgi = get_pgi(xtest_df, adv_lime, rfeature_1)
	# rfeature_2_pgi = get_pgi(xtest_df, adv_lime, rfeature_2)
	# rfeature_3_pgi = get_pgi(xtest_df, adv_lime, rfeature_3)
	# print(f'(R1) Mean {rfeature_1} PGI for adv_lime model: {np.mean(rfeature_1_pgi)}')
	# print(f'(R2) Mean {rfeature_2} PGI for adv_lime model: {np.mean(rfeature_2_pgi)}')
	# print(f'(R3) Mean {rfeature_3} PGI for adv_lime model: {np.mean(rfeature_3_pgi)}')

	# print(f'(L) SENS Mean {sens_feature} PGI for adv_lime model: {np.mean(sens_lime1_pgi)}')
	# print(f'(H) FOIL Mean {foil_feature1} PGI for adv_lime model: {np.mean(foil_lime1_pgi)}')

	# # adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(), discretize_continuous=False, categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two')])
                                               
	# # explanations = []
	# # for i in range(xtest.shape[0]):
	# # 	explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())

	# # # Display Results
	# # print ("\t# LIME Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
	# # print_experiment_summary(experiment_summary(explanations, features))
	# print ("Fidelity:", round(adv_lime.fidelity(xtest),2))

	# # Repeat the same thing for two features
	# adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain, feature_names=features, perturbation_multiplier=30, categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two')])
	# sens_lime2_pgi = get_pgi(xtest_df, adv_lime, sens_feature)
	# foil_lime21_pgi = get_pgi(xtest_df, adv_lime, foil_feature1)
	# foil_lime22_pgi = get_pgi(xtest_df, adv_lime, foil_feature2)

	# rfeature_1_pgi = get_pgi(xtest_df, adv_lime, rfeature_1)
	# rfeature_2_pgi = get_pgi(xtest_df, adv_lime, rfeature_2)
	# rfeature_3_pgi = get_pgi(xtest_df, adv_lime, rfeature_3)
	# print(f'(R1) Mean {rfeature_1} PGI for adv_lime model: {np.mean(rfeature_1_pgi)}')
	# print(f'(R2) Mean {rfeature_2} PGI for adv_lime model: {np.mean(rfeature_2_pgi)}')
	# print(f'(R3) Mean {rfeature_3} PGI for adv_lime model: {np.mean(rfeature_3_pgi)}')

	# print(f'(L) SENS Mean {sens_feature} PGI for adv_lime model: {np.mean(sens_lime2_pgi)}')
	# print(f'(H) FOIL Mean {foil_feature2} PGI for adv_lime model: {np.mean(foil_lime21_pgi)}')
	# print(f'(H) FOIL Mean {foil_feature2} PGI for adv_lime model: {np.mean(foil_lime22_pgi)}')

	# # adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(), discretize_continuous=False, categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two')])
                                               
	# # explanations = []
	# # for i in range(xtest.shape[0]):
	# # 	explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())	

	# # print ("\t# LIME Ranks and Pct Occurances two unrelated features:")
	# # print_experiment_summary(experiment_summary(explanations, features))
	# print ("Fidelity:", round(adv_lime.fidelity(xtest),2))
	# print ('---------------------')
	# print ('Beginning SHAP CC Experiments....')
	# print ('---------------------')

	# #Setup SHAP
	# background_distribution = shap.kmeans(xtrain,10)
	# adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features)
	# sens_shap1_pgi = get_pgi(xtest_df, adv_shap, sens_feature)
	# foil_shap1_pgi = get_pgi(xtest_df, adv_shap, foil_feature1)

	# rfeature_1_pgi = get_pgi(xtest_df, adv_lime, rfeature_1)
	# rfeature_2_pgi = get_pgi(xtest_df, adv_lime, rfeature_2)
	# rfeature_3_pgi = get_pgi(xtest_df, adv_lime, rfeature_3)
	# print(f'(R1) Mean {rfeature_1} PGI for adv_lime model: {np.mean(rfeature_1_pgi)}')
	# print(f'(R2) Mean {rfeature_2} PGI for adv_lime model: {np.mean(rfeature_2_pgi)}')
	# print(f'(R3) Mean {rfeature_3} PGI for adv_lime model: {np.mean(rfeature_3_pgi)}')

	# print(f'(L) SENS Mean {sens_feature} PGI for adv_lime model: {np.mean(sens_shap1_pgi)}')
	# print(f'(H) FOIL Mean {foil_feature1} PGI for adv_lime model: {np.mean(foil_shap1_pgi)}')

	# # adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
	# # explanations = adv_kerenel_explainer.shap_values(xtest)

	# # # format for display
	# # formatted_explanations = []
	# # for exp in explanations:
	# # 	formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

	# # print ("\t# SHAP Ranks and Pct Occurances one unrelated features:")
	# # print_experiment_summary(experiment_summary(formatted_explanations, features))
	# print ("Fidelity:",round(adv_shap.fidelity(xtest),2))

	# background_distribution = shap.kmeans(xtrain,10)
	# adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain, feature_names=features)
	# sens_shap2_pgi = get_pgi(xtest_df, adv_shap, sens_feature)
	# foil_shap21_pgi = get_pgi(xtest_df, adv_shap, foil_feature1)
	# foil_shap22_pgi = get_pgi(xtest_df, adv_shap, foil_feature2)

	# rfeature_1_pgi = get_pgi(xtest_df, adv_lime, rfeature_1)
	# rfeature_2_pgi = get_pgi(xtest_df, adv_lime, rfeature_2)
	# rfeature_3_pgi = get_pgi(xtest_df, adv_lime, rfeature_3)
	# print(f'(R1) Mean {rfeature_1} PGI for adv_lime model: {np.mean(rfeature_1_pgi)}')
	# print(f'(R2) Mean {rfeature_2} PGI for adv_lime model: {np.mean(rfeature_2_pgi)}')
	# print(f'(R3) Mean {rfeature_3} PGI for adv_lime model: {np.mean(rfeature_3_pgi)}')

	# print(f'(L) SENS Mean {sens_feature} PGI for adv_lime model: {np.mean(sens_shap2_pgi)}')
	# print(f'(H) FOIL Mean {foil_feature1} PGI for adv_lime model: {np.mean(foil_shap21_pgi)}')
	# print(f'(H) FOIL Mean {foil_feature2} PGI for adv_lime model: {np.mean(foil_shap22_pgi)}')
	
	# # adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
	# # explanations = adv_kerenel_explainer.shap_values(xtest)

	# # # format for display
	# # formatted_explanations = []
	# # for exp in explanations:
	# # 	formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

	# # print ("\t# SHAP Ranks and Pct Occurances two unrelated features:")
	# # print_experiment_summary(experiment_summary(formatted_explanations, features))
	# print ("Fidelity:",round(adv_shap.fidelity(xtest),2))
	# print ('---------------------')

if __name__ == "__main__":
	# experiment_main(True)
	experiment_main(False)
