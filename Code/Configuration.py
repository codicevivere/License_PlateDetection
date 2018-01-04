"""
Created on Tue Jan  5 13:35:56 2016

@author: sardendhu
"""

import os
import configparser as ConfigParser


def get_config_dir():
    dir_name = os.path.dirname(os.path.abspath(__file__))
    dir_name = os.path.abspath(os.path.join(dir_name, os.pardir))

    conf_name = 'config-local.conf'    
    dir_name = os.path.join(dir_name, conf_name)
    # print 'The directory where the config file is: ', dir_name
    return dir_name
    

def get_config():
    Config = ConfigParser.ConfigParser()   
    Config.read(get_config_dir())   
    # print ('All the configuration are: ', Config.sections())
    return Config


def get_datamodel_storage_path(): 
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    curr_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
    
    # print (curr_dir)

    conf = get_config()

    config_settings = {}
    config_settings["DataSet_LP"] = curr_dir+'/'+conf.get("Dataset", "DataSet_LP")
    config_settings["DataSet_NLP"] = curr_dir+'/'+conf.get("Dataset", "DataSet_NLP")
    

    config_settings["Data_feature_dir"] = curr_dir+'/'+conf.get("Feature-Labels", "Data_feature")
    config_settings["Class_labels_dir"] = curr_dir+'/'+conf.get("Feature-Labels", "Class_labels")
    config_settings["Data_feature_KernelCnvtd_dir"] = curr_dir+'/'+conf.get("Feature-Labels", "Data_feature_KernelCnvtd")
    config_settings["Data_feature_KernelCnvtd_tst_dir"] = curr_dir+'/'+conf.get("Feature-Labels", "Data_feature_KernelCnvtd_tst")
    config_settings["Theta_val_dir"] = curr_dir+'/'+conf.get("Feature-Labels", "Theta_val")

    config_settings["Linear_SVC_dir"] = curr_dir+'/'+conf.get("Models", "Linear_SVC")
    config_settings["SVM_RFB_dir"] = curr_dir+'/'+conf.get("Models","SVM_RFB")+'model_svm_rbf_%i_%i.pickle'
    config_settings["Models"] = curr_dir+'/'+conf.get("Models", "models")
    config_settings["ModelDir"] = curr_dir+'/'+conf.get("Models", "modeldir")
    config_settings["Selected_Model"] = curr_dir+'/'+conf.get("Models", "best_model_path")

    config_settings["Regions_of_Intrest"] = curr_dir+'/'+conf.get("Contored_images", "Regions_of_Intrest")
    config_settings["Extracted_license_plates"] = curr_dir+'/'+conf.get("Classified_License_plates", "classified_license_plate")

    config_settings["Images_to_classify"] = curr_dir+'/'+conf.get("Images_to_classify","Images_classify")
   

    config_settings["Transformed_LP"] = curr_dir+'/'+conf.get("Pic_transformed_path","transformed_LP")
    config_settings["Transformed_NLP"] = curr_dir+'/'+conf.get("Pic_transformed_path","transformed_NLP")
    return config_settings


# print (get_config_dir())
# print (get_datamodel_storage_path())