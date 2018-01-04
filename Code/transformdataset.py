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

from skimage.util import random_noise
from skimage import img_as_ubyte

conf = Configuration.get_datamodel_storage_path()







# ####################`
# #==============================================================================
# # 4: Find the Number plates of the vehicle
# #==============================================================================
# image_to_classify = "../image_2_classify (11).jpg"

# For classification let us use models one after another.
# For the cross validation data set the best model was. model_svm_rbf_10_1

def rotate_pics(input_path,output_path,borderlength=30):
	for file in glob.glob(input_path):
		image_orig = cv2.imread(file)
		image_orig=cv2.copyMakeBorder(image_orig,borderlength,borderlength,borderlength,borderlength,cv2.BORDER_CONSTANT,value=0)
		rows= image_orig.shape[0]
		cols= image_orig.shape[1]
		# print (file)
		image_name=os.path.basename(file)
		image_name=image_name.replace(" ", "")
		# print (image_name)
		for r in [-7,7]:
		# print ("hello")

			newname=image_name.split(".")[0]+"_r"+str(r)+".jpg"
			# print (newname)
			matrix= cv2.getRotationMatrix2D((cols/2,rows/2),r,1)
			transformed_im = cv2.warpAffine(image_orig,matrix,(cols,rows))
			# print(output_path+newname)
			cv2.imwrite(output_path+newname, transformed_im)


def translate_pics(input_path,output_path,borderlength=50):
	for file in glob.glob(input_path):
		image_orig = cv2.imread(file)
		image_orig=cv2.copyMakeBorder(image_orig,borderlength,borderlength,borderlength,borderlength,cv2.BORDER_CONSTANT,value=0)
		rows= image_orig.shape[0]
		cols= image_orig.shape[1]
		# print (file)
		image_name=os.path.basename(file)
		image_name=image_name.replace(" ", "")
		# print (image_name)
		for tx,ty in [[70,70],[-70,70]]:
			matrix= np.float32([[1,0,tx],[0,1,ty]])
		# print ("hello")

			newname=image_name.split(".")[0]+"_t"+str(tx)+"_"+str(ty)+".jpg"
			# print (newname)
			transformed_im = cv2.warpAffine(image_orig,matrix,(cols,rows))
			# print(output_path+newname)
			cv2.imwrite(output_path+newname, transformed_im)

def apply_affine(input_path,output_path,borderlength=30):
	for file in glob.glob(input_path):
		image_orig = cv2.imread(file)
		image_orig=cv2.copyMakeBorder(image_orig,borderlength,borderlength,borderlength,borderlength,cv2.BORDER_CONSTANT,value=0)
		rows= image_orig.shape[0]
		cols= image_orig.shape[1]
		# print (file)
		image_name=os.path.basename(file)
		image_name=image_name.replace(" ", "")
		# print (image_name)
		for r in [-7,0,7]:
		# print ("hello")
			matrixr= cv2.getRotationMatrix2D((cols/2,rows/2),r,1)
			for tx,ty in [[10,10],[-10,-10]]:
				matrixtr=matrixr
				matrixtr[0,2]=tx
				matrixtr[1,2]=ty
				newname=image_name.split(".")[0]+"_r"+str(r)+"_t"+str(tx)+"_"+str(ty)+".jpg"
				transformed_im = cv2.warpAffine(image_orig,matrixtr,(cols,rows))
				cv2.imwrite(output_path+newname, transformed_im)

def add_noise(input_path,output_path):
	for file in glob.glob(input_path):
		image_orig = cv2.imread(file)
		rows= image_orig.shape[0]
		cols= image_orig.shape[1]
		image_name=os.path.basename(file)
		image_name=image_name.replace(" ", "")

		newname=image_name.split(".")[0]+"_n"+".jpg"
		transformed_im=random_noise(image_orig, mode='pepper', seed=None, clip=True)
		transformed_im=img_as_ubyte(transformed_im, force_copy=False)
		# print(output_path+newname)
		cv2.imwrite(output_path+newname, transformed_im)

def smooth_image(input_path,output_path):
	for file in glob.glob(input_path):
		image_orig = cv2.imread(file)
		rows= image_orig.shape[0]
		cols= image_orig.shape[1]
		image_name=os.path.basename(file)
		image_name=image_name.replace(" ", "")

		newname=image_name.split(".")[0]+"_s"+".jpg"

		kernel = np.ones((5,5),np.float32)/25
		transformed_im = cv2.filter2D(image_orig,-1,kernel)
		# print(output_path+newname)
		cv2.imwrite(output_path+newname, transformed_im)

def apply_projective(input_path,output_path,borderlength=30):

	for file in glob.glob(input_path):
		image_orig = cv2.imread(file)
		image_orig=cv2.copyMakeBorder(image_orig,borderlength,borderlength,borderlength,borderlength,cv2.BORDER_CONSTANT,value=0)
		image_name=os.path.basename(file)
		image_name=image_name.replace(" ", "")
		rows= image_orig.shape[0]
		cols= image_orig.shape[1]
		matrixorig=np.zeros((4,2))
		matrixorig[0]=[0,0]
		matrixorig[1]=[0,cols-1]
		matrixorig[2]=[rows-1,cols-1]
		matrixorig[3]=[rows-1,0]

		matrix1=np.zeros((4,2))
		matrix1[0]=[0,0]
		matrix1[1]=[0,cols-1]
		matrix1[2]=[rows-1,0.75*cols-1]
		matrix1[3]=[rows-1,0.25*cols-1]

		tm = cv2.getPerspectiveTransform(matrixorig,matrix1)
		transformed_im =cv2.warpPerspective(image_orig,tm,(cols,rows))
		newname=image_name.split(".")[0]+"_a1"+".jpg"
		cv2.imwrite(output_path+newname, transformed_im)
		


__main__ = True

if __main__:
	# rotate_pics(conf['DataSet_LP'],conf['Transformed_LP'])
	# translate_pics(conf['DataSet_LP'],conf['Transformed_LP'])
	apply_affine(conf['DataSet_LP'],conf['Transformed_LP'])
	add_noise(conf['DataSet_LP'],conf['Transformed_LP'])
	smooth_image(conf['DataSet_LP'],conf['Transformed_LP'])

	apply_affine(conf['DataSet_NLP'],conf['Transformed_NLP'])
	add_noise(conf['DataSet_NLP'],conf['Transformed_NLP'])
	smooth_image(conf['DataSet_NLP'],conf['Transformed_NLP'])