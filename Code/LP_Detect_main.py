from __future__ import division

import os, glob
import numpy as np
import cv2
import Tools

import Configuration
from six.moves import cPickle as pickle
from sklearn.metrics import accuracy_score

from Classify import Classify
from Evaluate_models import Eval
from BldModel import SvmModel, Models
from Bld_FeatureCrps import CrtFeatures

from shutil import copyfile



conf = Configuration.get_datamodel_storage_path()







# ####################`
# #==============================================================================
# # 4: Find the Number plates of the vehicle
# #==============================================================================
# image_to_classify = "../image_2_classify (11).jpg"

# For classification let us use models one after another.
# For the cross validation data set the best model was. model_svm_rbf_10_1


def Extract_lisenceplates(model, extracted_license_plate_path):
	for num, image_inp in enumerate(glob.glob(conf['Images_to_classify']) ):
		print (image_inp)
		image_to_classify = cv2.imread(image_inp)
		# cv2.imshow('image',image_to_classify)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		image_resized = Tools.resize(image_to_classify, height=500)
		pred_dict = Classify().classify_new_instance(image_resized,model)
		# print (pred_dict)

		# print (model)
		probs = []
		coords=[]
		for image_fname,s_and_c in pred_dict.items():#range(0,len(pred_dict)):
			prob=s_and_c[1]
			x=s_and_c[2]
			y=s_and_c[3]
			w=s_and_c[4]
			h=s_and_c[5]
			probs.append(prob)
			coords.append([x,y,w,h])
		        # print (image_fname)
		probs = np.array(probs)
		coords = np.array(coords)
		ind = np.where(probs == np.max(probs))[0]


		filenames=pred_dict.keys()
		for i in ind:
			filename=filenames[i]
			copyfile(conf['Regions_of_Intrest']+filename, extracted_license_plate_path+filename.split(".")[0]+"_"+str(num)+".jpg")
			coord=coords[i]
			x=coord[0]
			y=coord[1]
			w=coord[2]
			h=coord[3]
			cv2.rectangle(image_resized,(x,y),(x+w,y+h),(0,255,0),2)


		cv2.namedWindow("Show")
		cv2.moveWindow("Show", 10, 50)
		cv2.imshow("Show",image_resized)
		cv2.waitKey()
		cv2.destroyAllWindows()
		# break




__main__ = True

if __main__:
	f = open(conf['Selected_Model'],'r')
	modelpath = f.read()
	# print(message)
	f.close()

	# model_path = os.path.dirname(os.path.abspath(modelpath))
	extracted_license_plate_path = conf["Extracted_license_plates"]
	model = pickle.load(open(modelpath, 'rb'))

	model.kernel=str(model.kernel)


	# create_feature_matrix()
	# train_model()
	# run_cross_valid()
	Extract_lisenceplates(model,extracted_license_plate_path)
