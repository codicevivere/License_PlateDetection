# Licence-Plate-Detection
The code is an approach to Detect licence plate of vehicles with use of Machine Learning Algorithms and Image Processing techniques

It uses HOG descriptor to calculate features of different contours in the image which are then fed into the SVM and the contour having highest score is selected to be the License Plate

Note:This code is built using python 2

The dataset has not been uploaded since special permissions are needed for sharing such data 
The models have been pretrained and the best one (with the highest F score ) is used by default for classifying images

For detecting License Plates in an image do the following:
	->Remove any files in Plate_detection/DataSet/Data-Files/images_to_classify/extracted_licenceplate_image
	->Remove any files in Plate_detection/DataSet/Data-Files/images_to_classify/contoured_images_roi
	->Remove any files in Plate_detection/DataSet/Data-Files/images_to_classify/images_classify 
	->Add image that needs to be classified into Plate_detection/DataSet/Data-Files/images_to_classify/images_classify 
	->Run Plate_detection/Code/LP_Detect_main.py

For training a new model:
	->Add the positive training images in Plate_detection/DataSet/Data-Files/images_dataset/Licence-Plate
	->Add the negative training images in Plate_detection/DataSet/Data-Files/images_dataset/Not-Licence-Plate
	->Run Plate_detection/Code/train_svm.py
