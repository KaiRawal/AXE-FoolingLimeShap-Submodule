"""
The experiment MAIN for GERMAN.
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

import lime
import lime.lime_tabular
import shap

from sklearn.cluster import KMeans 
import pandas as pd
from evals import get_pgi, get_bex, get_pgi_multi

from copy import deepcopy

import random
random.seed(42)

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
X, y, cols = get_and_preprocess_german(params)

features = [c for c in X]

sens_feature = 'Gender'
foil_feature = 'LoanRateAsPercentOfIncome'

gender_indc = features.index(sens_feature)
loan_rate_indc = features.index(foil_feature)

X = X.values

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1,random_state=42)
ss = StandardScaler().fit(xtrain)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)

mean_lrpi = np.mean(xtrain[:,loan_rate_indc])


categorical = ['Gender', 'ForeignWorker', 'Single', 'HasTelephone','CheckingAccountBalance_geq_0','CheckingAccountBalance_geq_200','SavingsAccountBalance_geq_100','SavingsAccountBalance_geq_500','MissedPayments','NoCurrentLoan','CriticalAccountOrLoansElsewhere','OtherLoansAtBank','OtherLoansAtStore','HasCoapplicant','HasGuarantor','OwnsHouse','RentsHouse','Unemployed','YearsAtCurrentJob_lt_1','YearsAtCurrentJob_geq_4','JobClassIsSkilled']
categorical = [features.index(c) for c in categorical]

xtest_df = pd.DataFrame(xtest, columns=features)

###
## The models f and psi for GERMAN.  We discriminate based on gender for f and consider loan rate % income for explanation
#

# the biased model 
class racist_model_f:
    # Decision rule: classify negative outcome if female
    def predict(self,X):
        return np.array([params.negative_outcome if x[gender_indc] < 0 else params.positive_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
    
# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to loan rate indc
    def predict_proba(self, X): 
        return one_hot_encode(np.array([params.negative_outcome if x[loan_rate_indc] > mean_lrpi else params.positive_outcome for x in X]))

##
###

def experiment_main(verbose=False):
	"""
	Run through experiments for LIME/SHAP on GERMAN.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""
	np.random.seed(42)
	rfeature_1, rfeature_2, rfeature_3 = random.sample(features, 3)
	
	if verbose:
		print ('---------------------')
		print ("Beginning LIME GERMAN Experiments....")
		print ("(These take some time to run because we have to generate explanations for every point in the test set) ")
		print ('---------------------')

	# Train the adversarial model for LIME with f and psi 
	adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features, perturbation_multiplier=30, categorical_features=categorical)
	if verbose:
		print(f'Model LIME trained with fidelity {adv_lime.fidelity(xtest)}')
	else:
		print(f'\t %german LIME-1 fidelity {adv_lime.fidelity(xtest)}')
		
	pgi_results = []
	bex_results = []
	for feature in features:
		if feature not in [sens_feature, foil_feature]:
			pgi_score = get_pgi(xtest_df, adv_lime, feature)
			bex_score = get_bex(xtest_df, adv_lime, feature)
			pgi_results.append((feature, np.mean(pgi_score)))
			bex_results.append((feature, np.mean(bex_score)))
	
	sens_lime_pgi = get_pgi(xtest_df, adv_lime, sens_feature)
	foil_lime_pgi = get_pgi(xtest_df, adv_lime, foil_feature)
	sens_lime_bex = get_bex(xtest_df, adv_lime, sens_feature)
	foil_lime_bex = get_bex(xtest_df, adv_lime, foil_feature)

	pgu_results = []
	for feature in features:
		inv_features = [f for f in features if f != feature]
		if feature not in [sens_feature, foil_feature]:
			pgu_score = 0-get_pgi_multi(xtest_df, adv_lime, inv_features)
			pgu_results.append((feature, np.mean(pgu_score)))
	sens_lime_pgu = 0-get_pgi_multi(xtest_df, adv_lime, [f for f in features if f != sens_feature])
	foil_lime_pgu = 0-get_pgi_multi(xtest_df, adv_lime, [f for f in features if f != foil_feature])
	
	if verbose:
		print(f'Mean PGI (random) for adv_lime model: {np.mean([score for _, score in pgi_results])}, Std Dev: {np.std([score for _, score in pgi_results])}')
		print(f'(L) SENS Mean {sens_feature} PGI for adv_lime model: {np.mean(sens_lime_pgi)}')
		print(f'(H) FOIL Mean {foil_feature} PGI for adv_lime model: {np.mean(foil_lime_pgi)}')
		print(f'Mean BEX (random) for adv_lime model: {np.mean([score for _, score in bex_results])}, Std Dev: {np.std([score for _, score in bex_results])}')
		print(f'(L) SENS Mean {sens_feature} BEX for adv_lime model: {np.mean(sens_lime_bex)}')
		print(f'(H) FOIL Mean {foil_feature} BEX for adv_lime model: {np.mean(foil_lime_bex)}')
	else:
		print(f' german & lime & PGI & {np.mean(sens_lime_pgi):.3f} & {np.mean(foil_lime_pgi):.3f} & na & {np.mean([score for _, score in pgi_results]):.3f} \\\\ ')
		print(f' german & lime & PGU & {np.mean(sens_lime_pgu):.3f} & {np.mean(foil_lime_pgu):.3f} & na & {np.mean([score for _, score in pgu_results]):.3f} \\\\ ')
		print(f' german & lime & AXE & {np.mean(sens_lime_bex):.3f} & {np.mean(foil_lime_bex):.3f} & na & {np.mean([score for _, score in bex_results]):.3f} \\\\ ')

	background_distribution = KMeans(n_clusters=10,random_state=0).fit(xtrain).cluster_centers_
	adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, 
			feature_names=features, background_distribution=background_distribution, rf_estimators=100, n_samples=5e4)
	if verbose:
		print(f'Model SHAP trained with fidelity {adv_shap.fidelity(xtest)}')
	else:
		print(f'\t %german SHAP-1 fidelity {adv_shap.fidelity(xtest)}')
	
	pgi_results_shap = []
	bex_results_shap = []
	for feature in features:
		if feature not in [sens_feature, foil_feature]:
			pgi_score_shap = get_pgi(xtest_df, adv_shap, feature)
			bex_score_shap = get_bex(xtest_df, adv_shap, feature)
			pgi_results_shap.append((feature, np.mean(pgi_score_shap)))
			bex_results_shap.append((feature, np.mean(bex_score_shap)))
	
	sens_shap_pgi = get_pgi(xtest_df, adv_shap, sens_feature)
	foil_shap_pgi = get_pgi(xtest_df, adv_shap, foil_feature)
	sens_shap_bex = get_bex(xtest_df, adv_shap, sens_feature)
	foil_shap_bex = get_bex(xtest_df, adv_shap, foil_feature)

	pgu_results = []
	for feature in features:
		inv_features = [f for f in features if f != feature]
		if feature not in [sens_feature, foil_feature]:
			pgu_score = 0-get_pgi_multi(xtest_df, adv_shap, inv_features)
			pgu_results.append((feature, np.mean(pgu_score)))
	sens_shap_pgu = 0-get_pgi_multi(xtest_df, adv_shap, [f for f in features if f != sens_feature])
	foil_shap_pgu = 0-get_pgi_multi(xtest_df, adv_shap, [f for f in features if f != foil_feature])
		
	if verbose:
		print(f'Mean PGI (random) for adv_shap model: {np.mean([score for _, score in pgi_results_shap])}, Std Dev: {np.std([score for _, score in pgi_results_shap])}')
		print(f'(L) SENS Mean {sens_feature} PGI for adv_shap model: {np.mean(sens_shap_pgi)}')
		print(f'(H) FOIL Mean {foil_feature} PGI for adv_shap model: {np.mean(foil_shap_pgi)}')
		print(f'Mean BEX (random) for adv_shap model: {np.mean([score for _, score in bex_results_shap])}, Std Dev: {np.std([score for _, score in bex_results_shap])}')
		print(f'(L) SENS Mean {sens_feature} BEX for adv_shap model: {np.mean(sens_shap_bex)}')
		print(f'(H) FOIL Mean {foil_feature} BEX for adv_shap model: {np.mean(foil_shap_bex)}')
	else:
		print(f' german & shap & PGI & {np.mean(sens_shap_pgi):.3f} & {np.mean(foil_shap_pgi):.3f} & na & {np.mean([score for _, score in pgi_results_shap]):.3f} \\\\ ')
		print(f' german & shap & PGU & {np.mean(sens_shap_pgu):.3f} & {np.mean(foil_shap_pgu):.3f} & na & {np.mean([score for _, score in pgu_results]):.3f} \\\\ ')
		print(f' german & shap & AXE & {np.mean(sens_shap_bex):.3f} & {np.mean(foil_shap_bex):.3f} & na & {np.mean([score for _, score in bex_results_shap]):.3f} \\\\ ')

		

	return

	# sens_lime_pgi = get_pgi(xtest_df, adv_lime, sens_feature)
	# foil_lime_pgi = get_pgi(xtest_df, adv_lime, foil_feature)

	# rfeature_1_pgi = get_pgi(xtest_df, adv_lime, rfeature_1)
	# rfeature_2_pgi = get_pgi(xtest_df, adv_lime, rfeature_2)
	# rfeature_3_pgi = get_pgi(xtest_df, adv_lime, rfeature_3)
	# print(f'(R1) Mean {rfeature_1} PGI for adv_lime model: {np.mean(rfeature_1_pgi)}')
	# print(f'(R2) Mean {rfeature_2} PGI for adv_lime model: {np.mean(rfeature_2_pgi)}')
	# print(f'(R3) Mean {rfeature_3} PGI for adv_lime model: {np.mean(rfeature_3_pgi)}')

	# print(f'(L) SENS Mean {sens_feature} PGI for adv_lime model: {np.mean(sens_lime_pgi)}')
	# print(f'(H) FOIL Mean {foil_feature} PGI for adv_lime model: {np.mean(foil_lime_pgi)}')
	
	# # adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(), discretize_continuous=False, categorical_features=categorical)
                                               
	# # explanations = []	
	# # for i in range(xtest.shape[0]):
	# # 	explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())
	# # 	# explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba, num_features=len(xtest[i])).as_list())

	# # # # Save the features to a file
	# # # with open('data/INTERIMS/german/features.txt', 'w') as f:
	# # # 	f.write(f'{str(features)}')
	# # # 	f.write('\n')

	# # # explanations_df = pd.DataFrame([{feature.split('=')[0]: value for feature, value in exp} for exp in explanations])
	# # # explanations_df.to_csv('data/INTERIMS/german/lime/explanations.csv', index=False)


	# # # np.save('data/INTERIMS/german/lime/xtest.npy', xtest)
	# # # np.save('data/INTERIMS/german/lime/xtrain.npy', xtrain)
	
	# # # y_pred_proba_test = adv_lime.predict_proba(xtest)
	# # # y_pred_test = adv_lime.predict(xtest)
	
	# # # np.save('data/INTERIMS/german/lime/y_pred_proba_test.npy', y_pred_proba_test)
	# # # np.save('data/INTERIMS/german/lime/y_pred_test.npy', y_pred_test)
	# # # np.save('data/INTERIMS/german/lime/y_test.npy', ytest)

	# # # y_pred_proba_train = adv_lime.predict_proba(xtrain)
	# # # y_pred_train = adv_lime.predict(xtrain)
	
	# # # np.save('data/INTERIMS/german/lime/y_pred_proba_train.npy', y_pred_proba_train)
	# # # np.save('data/INTERIMS/german/lime/y_pred_train.npy', y_pred_train)
	# # # np.save('data/INTERIMS/german/lime/y_train.npy', ytrain)

	# # # Display Results
	# # print ("\t# LIME Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
	# # print_experiment_summary(experiment_summary(explanations, features))
	# print ("Fidelity:", round(adv_lime.fidelity(xtest),2))


	
	# print ('---------------------')
	# print ('Beginning SHAP GERMAN Experiments....')
	# print ('---------------------')

	# #Setup SHAP
	# background_distribution = KMeans(n_clusters=10,random_state=0).fit(xtrain).cluster_centers_
	# adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, 
	# 		feature_names=features, background_distribution=background_distribution, rf_estimators=100, n_samples=5e4)
	# sens_shap_pgi = get_pgi(xtest_df, adv_shap, sens_feature)
	# foil_shap_pgi = get_pgi(xtest_df, adv_shap, foil_feature)

	# rfeature_1_pgi = get_pgi(xtest_df, adv_shap, rfeature_1)
	# rfeature_2_pgi = get_pgi(xtest_df, adv_shap, rfeature_2)
	# rfeature_3_pgi = get_pgi(xtest_df, adv_shap, rfeature_3)
	# print(f'(R1) Mean {rfeature_1} PGI for adv_shap model: {np.mean(rfeature_1_pgi)}')
	# print(f'(R2) Mean {rfeature_2} PGI for adv_shap model: {np.mean(rfeature_2_pgi)}')
	# print(f'(R3) Mean {rfeature_3} PGI for adv_shap model: {np.mean(rfeature_3_pgi)}')


	# print(f'Mean {sens_feature} PGI for adv_shap model: {np.mean(sens_shap_pgi)}')
	# print(f'Mean {foil_feature} PGI for adv_shap model: {np.mean(foil_shap_pgi)}')
	
	# # adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution,)
	# # explanations = adv_kerenel_explainer.shap_values(xtest)

	# # # format for display
	# # formatted_explanations = []
	# # for exp in explanations:
	# # 	formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

	# # print ("\t# SHAP Ranks and Pct Occurances one unrelated features:")
	
	# # print_experiment_summary(experiment_summary(formatted_explanations, features))

	# print ("Fidelity:",round(adv_shap.fidelity(xtest),2))

	# print ('---------------------')

if __name__ == "__main__":
	# experiment_main(True)
	experiment_main(False)
