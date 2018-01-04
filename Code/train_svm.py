from __future__ import division


import random
import os, glob
import numpy as np
import cv2
import Tools

import Configuration
from six.moves import cPickle as pickle 
from sklearn.metrics import f1_score

from Classify import Classify
from Evaluate_models import Eval
from BldModel import SvmModel, Models
from Bld_FeatureCrps import CrtFeatures

from shutil import copyfile



conf = Configuration.get_datamodel_storage_path()


#==============================================================================
# 1: create and store features and labels of the images: (use Bld_FeatureCrps)
#==============================================================================
def create_feature_matrix():
	print ("creating feature matrix--------------\n")
	CrtFeatures().create_dataset_features(store='Yes')


#==============================================================================
# 2: Train the model and store it into disk for future use.
#==============================================================================

# Read the features stored on the disk
def split_into_categories():
	print ("splitting and randomizing data--------------\n")
	features = np.genfromtxt(conf['Data_feature_dir'], dtype=float, delimiter=',')
	labels = np.genfromtxt(conf['Class_labels_dir'], dtype=float, delimiter=',')
	numimgs=features.shape[0]

	inds=range(0,numimgs)
	random.shuffle(inds)
	# print (inds)
	features=features[inds]
	labels=labels[inds]
	ntrain=int(0.6*numimgs)
	ncrossvalidate=int(0.2*numimgs)
	ntest=numimgs-ntrain- ncrossvalidate
	
	trainf=features[0:ntrain,:]
	trainl=labels[0:ntrain]

	crvf=features[ntrain:ntrain+ncrossvalidate,:]
	crvl=labels[ntrain:ntrain+ncrossvalidate]

	testf=features[ntrain+ncrossvalidate:,:]
	testl=labels[ntrain+ncrossvalidate::]

	return trainf,trainl,crvf,crvl,testf,testl

#####training svm
def train_models(train_features,train_labels):
	

	sigma=4
	max_iter=5000
	alpha=0.003
	c=10

	'''
    MODEL 1: find the Kernel cnvrt features, cost_function and Parameters (thetas) and store them as models
	Note:
		The more the sigma is the less the gamma will be
	'''
	# features_krnl_cnvrt,j_theta, theta = SvmModel().main_call(sigma, features, labels, max_iter, alpha, c)
	# np.savetxt(conf["Data_feature_KernelCnvtd_dir"], features_krnl_cnvrt, delimiter=",")
	# np.savetxt(conf["Theta_val_dir"], theta, delimiter=",")

	# '''
	#     MODEL 2: Use packages models, linearSvc, and rbf's with varying gamma and c values.
	# '''
	Models(type = 'rbf').fit(train_features, train_labels)




# #==============================================================================
# # 3: Use the patameters and Operate on cross validation dataset
# #==============================================================================
 #####finding accuracy
def predict_on_set(features, model=None):
	feature_matrix_valid = np.array(features, dtype="float64")
	if model=='rbf':
		prediction_dict = Eval().test_using_model_rbf(feature_matrix_valid)
	elif model=='self':
		prediction_dict = Eval().test_using_model_self(feature_matrix_valid)
	else:
		print ('You should specify the model in which you would wanna crossvalidate your data')
	return prediction_dict

def run_cross_valid(crvf,crvl,testf,testl):
	print ('Running classification on cross validation set:::::------\n')
	prediction_dict = predict_on_set(crvf, model='rbf')
	maxacc=0
	maxacc_model=None
	for model, pred in prediction_dict.items():
		accuracy=f1_score(crvl, pred)
		print ('The accuracy of model %s is: '%model, accuracy)
		if(accuracy>maxacc):
			maxacc=accuracy
			maxacc_model=model
	print ('-------------best model selected  is: ')
	print (maxacc_model)
	print ('\n')

	pred,modelpath=Eval().test_set_on_specific_rbf_model(testf, maxacc_model)
	accuracy=f1_score(testl, pred)
	print ('The accuracy of this model on test set is: ', accuracy)

	return modelpath





# ####################`
# #==============================================================================
# # 4: Find the Number plates of the vehicle
# #==============================================================================
# image_to_classify = "../image_2_classify (11).jpg"

# For classification let us use models one after another.
# For the cross validation data set the best model was. model_svm_rbf_10_1

def train_and_evaluate_calssifiers():
	create_feature_matrix()
	trainf,trainl,crvf,crvl,testf,testl=split_into_categories()
	train_models(trainf,trainl)
	modelpath=run_cross_valid(crvf,crvl,testf,testl)
	f = open(conf['Selected_Model'],'w')
	f.write(modelpath)
	f.close()


__main__ = True

if __main__:

	train_and_evaluate_calssifiers()
	
	
