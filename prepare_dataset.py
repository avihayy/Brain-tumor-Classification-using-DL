"""
prepere_dataset.py contains all the functions needed for prepare the dataset for train validation and test, include:
1) separating the dataset into train validation and test according to random approach or subject(patient) approach.
2) reading the images and there masks from .mat files and resizing them to 256x256.
"""


import numpy as np
import h5py
import os
from pathlib import Path
import shutil
import cv2
import random


def seperate_datasets(all_mat_file_dir_path):
    tumor_dataset_path=Path(all_mat_file_dir_path)
    if not os.path.isdir(tumor_dataset_path.parent/ "label_1"):
        os.mkdir(tumor_dataset_path.parent/ "label_1")
    else:
        shutil.rmtree(tumor_dataset_path.parent / "label_1")
        os.mkdir(tumor_dataset_path.parent / "label_1")

    if not os.path.isdir(tumor_dataset_path.parent/ "label_2"):
        os.mkdir(tumor_dataset_path.parent / "label_2")
    else:
        shutil.rmtree(tumor_dataset_path.parent/ "label_2")
        os.mkdir(tumor_dataset_path.parent/ "label_2")

    if not os.path.isdir(tumor_dataset_path.parent/ "label_3"):
        os.mkdir(tumor_dataset_path.parent / "label_3")
    else:
        shutil.rmtree(tumor_dataset_path.parent / "label_3")
        os.mkdir(tumor_dataset_path.parent/ "label_3")

    all_dir_files=os.listdir(all_mat_file_dir_path)
    for h, file_name in enumerate(np.sort(all_dir_files)):
        if file_name[0] == '.':
            continue
        file_path=all_mat_file_dir_path+"/"+file_name
        f = h5py.File(file_path, 'r')
        data = f.get('cjdata')
        label = int(data.get('label').value[0][0])
        PID = data.get('PID').value
        PID ="".join([str(chr(item)) for item in PID])
        destination_dir = "label_{}".format(label)
        destination_dir_PID = destination_dir+"/"+PID
        destination_dir_all_img = destination_dir + "/" + "all_images"
        if not os.path.isdir(tumor_dataset_path.parent / destination_dir_PID):
            os.mkdir(tumor_dataset_path.parent / destination_dir_PID)
        if not os.path.isdir(tumor_dataset_path.parent / destination_dir_all_img):
            os.mkdir(tumor_dataset_path.parent / destination_dir_all_img)
        destination_path_PID = tumor_dataset_path.parent/destination_dir_PID
        destination_path_all_img = tumor_dataset_path.parent / destination_dir_all_img
        shutil.copy(file_path, destination_path_PID)
        shutil.copy(file_path, destination_path_all_img)

def seperate_dataset_for_validation(tumor_dataset_dir_path,validation_percentage,how_to_seperate,seed):
    np.random.seed(seed)
    dest_dir = tumor_dataset_dir_path + "/validation"
    dest_dir_label_1_2 = dest_dir + "/label_1_2"
    dest_dir_label_1_3 = dest_dir + "/label_1_3"
    dest_dir_label_2_3 = dest_dir + "/label_2_3"
    dest_dir_all_labels = dest_dir + "/all_labels"
    dest_dir_label_list = [dest_dir_label_1_2, dest_dir_label_1_3, dest_dir_label_2_3, dest_dir_all_labels]
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    else:
        shutil.rmtree(dest_dir)
        os.mkdir(dest_dir)
    for i in range(len(dest_dir_label_list)):
        if not os.path.isdir(dest_dir_label_list[i]):
            os.mkdir(dest_dir_label_list[i])
        else:
            shutil.rmtree(dest_dir_label_list[i])
            os.mkdir(dest_dir_label_list[i])

    label_1_path = tumor_dataset_dir_path + "/label_1"
    label_2_path = tumor_dataset_dir_path + "/label_2"
    label_3_path = tumor_dataset_dir_path + "/label_3"

    all_label_path_list = [label_1_path, label_2_path, label_3_path]
    image_for_test = []
    for i, dir_path in enumerate(all_label_path_list):
        all_dirs_in_curr_label=np.flip(np.sort(os.listdir(dir_path)))
        num_of_images_in_curr_label = len(os.listdir(dir_path + "/all_images"))
        num_of_val_img_i_need_to_take = int(num_of_images_in_curr_label * validation_percentage)
        if how_to_seperate == "random":
            all_label_images = np.sort(os.listdir(dir_path + "/all_images"))
            p = np.random.permutation(len(all_label_images))
            curr_label_images_for_val = np.array(all_label_images)[p[:num_of_val_img_i_need_to_take]]
            curr_label_images_for_val = [dir_path+"/all_images/"+file for file in curr_label_images_for_val]
        else:
            curr_label_images_for_val = []
            how_many_images_i_already_took = 0
            for h, patient_dir in enumerate(all_dirs_in_curr_label):
                if patient_dir == "all_images":
                    continue
                if patient_dir[0] == '.':
                    continue
                patient_dir_path = dir_path + "/" + patient_dir
                all_curr_patient_files = os.listdir(patient_dir_path)
                num_of_curr_patient_images = len(all_curr_patient_files)
                if (num_of_curr_patient_images + how_many_images_i_already_took) > num_of_val_img_i_need_to_take:
                    break
                else:
                    for file in all_curr_patient_files:
                        if file[0] == '.':
                            continue
                        curr_label_images_for_val.append(patient_dir_path+"/"+file)
                        how_many_images_i_already_took = how_many_images_i_already_took + 1
        image_for_test.append(curr_label_images_for_val)

    for i in range(1, len(image_for_test) + 1):
        curr_test_files_list = image_for_test[i - 1]
        for file_path in curr_test_files_list:
            if os.path.isdir(file_path):
                continue
            if i == 1:
                shutil.copy(file_path, dest_dir_label_1_2)
                shutil.copy(file_path, dest_dir_label_1_3)
            if i == 2:
                shutil.copy(file_path, dest_dir_label_1_2)
                shutil.copy(file_path, dest_dir_label_2_3)
            if i == 3:
                shutil.copy(file_path, dest_dir_label_1_3)
                shutil.copy(file_path, dest_dir_label_2_3)
            if how_to_seperate != "random":
                os.remove(file_path)
                split_file_path=file_path.split('/')
                split_file_path[2]="/all_images"
                file_path="/".join(split_file_path)
            shutil.move(file_path, dest_dir_all_labels)

def seperate_dataset_for_testing(tumor_dataset_dir_path, test_percentage,how_to_seperate,seed):
    np.random.seed(seed)
    dest_dir = tumor_dataset_dir_path + "/test"
    dest_dir_label_1_2 = dest_dir + "/label_1_2"
    dest_dir_label_1_3 = dest_dir + "/label_1_3"
    dest_dir_label_2_3 = dest_dir + "/label_2_3"
    dest_dir_all_labels = dest_dir + "/all_labels"
    dest_dir_label_list =[dest_dir_label_1_2,dest_dir_label_1_3,dest_dir_label_2_3,dest_dir_all_labels]
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    else:
        shutil.rmtree(dest_dir)
        os.mkdir(dest_dir)
    for i in range(len(dest_dir_label_list)):
        if not os.path.isdir(dest_dir_label_list[i]):
            os.mkdir(dest_dir_label_list[i])
        else:
            shutil.rmtree(dest_dir_label_list[i])
            os.mkdir(dest_dir_label_list[i])
    label_1_path = tumor_dataset_dir_path + "/label_1"
    label_2_path = tumor_dataset_dir_path + "/label_2"
    label_3_path = tumor_dataset_dir_path + "/label_3"
    all_label_path_list=[label_1_path,label_2_path,label_3_path]
    image_for_test=[]
    for i,dir_path in enumerate(all_label_path_list):
        all_dirs_in_curr_label = np.flip(np.sort(os.listdir(dir_path)))
        num_of_images_in_curr_label=len(os.listdir(dir_path+"/all_images"))
        num_of_test_img_i_need_to_take=int(num_of_images_in_curr_label*test_percentage)
        if how_to_seperate == "random":
            all_label_images=np.sort(os.listdir(dir_path+"/all_images"))
            p = np.random.permutation(len(all_label_images))
            curr_label_images_for_test = np.array(all_label_images)[p[:num_of_test_img_i_need_to_take]]
            curr_label_images_for_test = [dir_path+"/all_images/"+file for file in curr_label_images_for_test]
        else:
            curr_label_images_for_test = []
            how_many_images_i_already_took = 0
            for h, patient_dir in enumerate(all_dirs_in_curr_label):
                if patient_dir == "all_images":
                    continue
                if patient_dir[0] == '.':
                    continue

                patient_dir_path = dir_path + "/" + patient_dir
                all_curr_patient_files = os.listdir(patient_dir_path)
                num_of_curr_patient_images=len(all_curr_patient_files)
                if (num_of_curr_patient_images+how_many_images_i_already_took)>num_of_test_img_i_need_to_take:
                    break
                else:
                    for file in all_curr_patient_files:
                        if file[0] == '.':
                            continue
                        curr_label_images_for_test.append(patient_dir_path+"/"+file)
                        how_many_images_i_already_took = how_many_images_i_already_took +1

        image_for_test.append(curr_label_images_for_test)

    for i in range(1, len(image_for_test) + 1):
        curr_test_files_list = image_for_test[i - 1]
        for file_path in curr_test_files_list:
            if os.path.isdir(file_path):
                continue
            if i == 1:
                shutil.copy(file_path, dest_dir_label_1_2)
                shutil.copy(file_path, dest_dir_label_1_3)
            if i == 2:
                shutil.copy(file_path, dest_dir_label_1_2)
                shutil.copy(file_path, dest_dir_label_2_3)
            if i == 3:
                shutil.copy(file_path, dest_dir_label_1_3)
                shutil.copy(file_path, dest_dir_label_2_3)
            if how_to_seperate != "random":
                os.remove(file_path)
                split_file_path=file_path.split('/')
                split_file_path[2]="/all_images"
                file_path="/".join(split_file_path)
            shutil.move(file_path, dest_dir_all_labels)




def read_data_from_dir(mat_file_dir_path,img_size):

    all_files_in_dir=np.sort(os.listdir(mat_file_dir_path))
    tumor_image_list=[]
    tumor_mask_list=[]
    tumor_label_list=[]
    all_label_pid_list=[]
    for i,filename in enumerate(all_files_in_dir):
        if filename[0] == '.':
            continue
        file_path = mat_file_dir_path + "/" + filename
        f = h5py.File(file_path, 'r')
        data = f.get('cjdata')
        PID = (data.get('PID').value).reshape(-1)
        PID = "".join([str(chr(item)) for item in PID])
        mask = data.get('tumorMask')
        image = data.get('image')
        lab_temp = data.get('label')
        img_data=image.value
        mask_data = mask.value
        label=lab_temp.value[0][0]
        img_data_resize=cv2.resize(img_data,img_size)
        mask_data_resize = cv2.resize(mask_data,img_size)
        img_data_resize=img_data_resize/np.max(img_data_resize)
        mask_data_resize = mask_data_resize / np.max(mask_data_resize)
        tumor_image_list.append(img_data_resize.reshape(img_data_resize.shape+(1,)))
        tumor_mask_list.append(mask_data_resize.reshape(mask_data_resize.shape+(1,)))
        all_label_pid_list.append(PID)
        tumor_label_list.append(label)

    return tumor_image_list,tumor_mask_list,tumor_label_list,all_label_pid_list



def prepare_data_for_train(data_type,img_size,dataset_num,how_to_separate):
    from tumor_classification_NN import unet_model
    if data_type=="label_1_2":
        mat_file_dir_path_label_first = "tumor_dataset/train_set/train_" + how_to_separate + "_separation/label_1/all_images/{}".format(dataset_num)
        mat_file_dir_path_label_second = "tumor_dataset/train_set/train_" + how_to_separate + "_separation/label_2/all_images/{}".format(dataset_num)
    if data_type == "label_1_3":
        mat_file_dir_path_label_first = "tumor_dataset/train_set/train_" + how_to_separate + "_separation/label_1/all_images/{}".format(dataset_num)
        mat_file_dir_path_label_second = "tumor_dataset/train_set/train_" + how_to_separate + "_separation/label_3/all_images/{}".format(dataset_num)
    if data_type == "label_2_3":
        mat_file_dir_path_label_first = "tumor_dataset/train_set/train_" + how_to_separate + "_separation/label_2/all_images/{}".format(dataset_num)
        mat_file_dir_path_label_second = "tumor_dataset/train_set/train_" + how_to_separate + "_separation/label_3/all_images/{}".format(dataset_num)

    tumor_image_list_first, tumor_mask_list_first, tumor_label_list_pre_first, patient_id_list_first = read_data_from_dir(mat_file_dir_path_label_first, img_size)
    tumor_image_list_second, tumor_mask_list_second, tumor_label_list_pre_second, patient_id_list_second = read_data_from_dir(mat_file_dir_path_label_second,img_size)
    tumor_image_arr=np.concatenate((tumor_image_list_first,tumor_image_list_second))
    tumor_GT_mask_arr = np.concatenate((tumor_mask_list_first,tumor_mask_list_second))

    segmentation_model = unet_model(img_size+(1,),use_attention=True)
    segmentation_model.load_weights("saved_classifier_model/unet_segmentation_model_" + how_to_separate + "_separation.h5")
    pred_masks_arr = segmentation_model.predict(tumor_image_arr)
    pred_masks_arr=pred_masks_arr.round()
    patient_id_list=np.unique(np.concatenate((patient_id_list_first,patient_id_list_second)))
    print("##########################################################################")
    print("patient id which trained(total {}):{}".format(len(patient_id_list), patient_id_list))
    print("##########################################################################")

    if data_type == "label_1_2":
        tumor_label_arr_first = np.copy((np.array(tumor_label_list_pre_first) - 1.0))
        tumor_label_arr_second = np.copy((np.array(tumor_label_list_pre_second) - 1.0))
    if data_type == "label_1_3":
        tumor_label_arr_first = np.copy((np.array(tumor_label_list_pre_first) - 1.0))
        tumor_label_arr_second = np.copy((np.array(tumor_label_list_pre_second) - 2.0))
    if data_type == "label_2_3":
        tumor_label_arr_first = np.copy((np.array(tumor_label_list_pre_first) - 2.0))
        tumor_label_arr_second = np.copy((np.array(tumor_label_list_pre_second) - 2.0))
    tumor_label_arr = np.concatenate((tumor_label_arr_first, tumor_label_arr_second))
    x_img=np.array([np.concatenate((i,i,i),axis=2) for i in tumor_image_arr])
    x_mask=np.array([np.concatenate((i,i,i),axis=2) for i in pred_masks_arr])
    y=tumor_label_arr.astype(int)
    p = np.random.permutation(x_img.shape[0])
    x_after_shuffle = x_img[p]
    x_mask_after_shuffle = x_mask[p]
    y_after_shuffle = y[p]
    return x_after_shuffle,x_mask_after_shuffle,y_after_shuffle


def prepare_data_for_validation(img_size,data_type,how_to_separate):
    val_dir = "tumor_dataset/validation_set/validation_" + how_to_separate + "_separation/" + data_type
    tumor_image_list_val, tumor_mask_list_val, tumor_label_list_pre_val, patient_id_list = read_data_from_dir(val_dir, img_size)
    print("##########################################################################")
    print("patient id which in validation(total {}):{}".format(len(np.unique(patient_id_list)), np.unique(patient_id_list)))
    print("##########################################################################")
    if data_type == "label_1_2":
        tumor_label_arr = np.copy((np.array(tumor_label_list_pre_val) - 1.0))
    if data_type == "label_1_3":
        tumor_label_arr = np.zeros_like(tumor_label_list_pre_val)
        for i in range(len(tumor_label_list_pre_val)):
            if tumor_label_list_pre_val[i] == 3:
                tumor_label_arr[i] = tumor_label_list_pre_val[i] - 2.0
    if data_type == "label_2_3":
        tumor_label_arr = np.copy((np.array(tumor_label_list_pre_val) - 2.0))

    x_val= np.array([np.concatenate((i, i, i), axis=2) for i in tumor_image_list_val])
    x_val_mask = np.array([np.concatenate((i, i, i), axis=2) for i in tumor_mask_list_val])
    y_val = tumor_label_arr.astype(int)
    return x_val,x_val_mask, y_val


def prepare_data_for_test(img_size,data_type,how_to_separate):

    test_dir = "tumor_dataset/test_set/test_"+how_to_separate+"_separation/" + data_type
    tumor_image_list_test, tumor_mask_list_test, tumor_label_list_pre_test, patient_id_list = read_data_from_dir(test_dir, img_size)
    print("##########################################################################")
    print("patient id which tested(total {}):{}".format(len(np.unique(patient_id_list)),np.unique(patient_id_list)))
    print("##########################################################################")
    if data_type == "label_1_2":
        tumor_label_arr = np.copy((np.array(tumor_label_list_pre_test) - 1.0))
    if data_type == "label_1_3":
        tumor_label_arr=np.zeros_like(tumor_label_list_pre_test)
        for i in range(len(tumor_label_list_pre_test)):
            if tumor_label_list_pre_test[i] == 3:
                tumor_label_arr[i] = tumor_label_list_pre_test[i]-2.0
    if data_type == "label_2_3":
        tumor_label_arr = np.copy((np.array(tumor_label_list_pre_test) - 2.0))
    if data_type == "all_labels":
        tumor_label_arr = np.copy((np.array(tumor_label_list_pre_test) - 1.0))

    x_test = np.array([np.concatenate((i, i, i), axis=2) for i in tumor_image_list_test])
    x_mask_test = np.array([np.concatenate((i, i, i), axis=2) for i in tumor_mask_list_test])
    y_test = tumor_label_arr.astype(int)
    return x_test,x_mask_test, y_test


def split_dataset_for_colab_training(tumor_dataset_dir_path,N,seed):
    random.seed(seed)
    label_1_path = tumor_dataset_dir_path+"/label_1/all_images"
    label_2_path = tumor_dataset_dir_path + "/label_2/all_images"
    label_3_path = tumor_dataset_dir_path + "/label_3/all_images"
    path_arr=[label_1_path,label_2_path,label_3_path]
    for i in range(len(path_arr)):

        all_files_in_curr_dir = np.sort(os.listdir(path_arr[i]))
        random.shuffle(all_files_in_curr_dir)
        num_of_file_in_dir=len(all_files_in_curr_dir)
        num_of_files_in_specific_dir=int(num_of_file_in_dir/N)

        for j in range(N-1):
            specific_dir_path=path_arr[i] + "/{}".format(j+1)
            if os.path.isdir(specific_dir_path):
                shutil.rmtree(specific_dir_path)
            os.mkdir(specific_dir_path)

            cut_off_files=all_files_in_curr_dir[num_of_files_in_specific_dir*j:num_of_files_in_specific_dir*(j+1)]
            for file in cut_off_files:
                if os.path.isfile(path_arr[i]+"/"+file):
                    shutil.move(path_arr[i]+"/"+file,specific_dir_path)

        specific_dir_path = path_arr[i] + "/{}".format(N)
        if os.path.isdir(specific_dir_path):
            shutil.rmtree(specific_dir_path)
        os.mkdir(specific_dir_path)

        all_remind_files_in_curr_dir = np.sort(os.listdir(path_arr[i]))
        for file in all_remind_files_in_curr_dir:
            if os.path.isfile(path_arr[i] + "/" + file):
                if os.path.isfile(path_arr[i] + "/" + file):
                    shutil.move(path_arr[i] + "/" + file, specific_dir_path)


def create_train_dataset_for_unet(tumor_dataset_dir_path,num_of_subdir,dest_dir_path):
    label_1_path = tumor_dataset_dir_path+"/label_1/all_images"
    label_2_path = tumor_dataset_dir_path + "/label_2/all_images"
    label_3_path = tumor_dataset_dir_path + "/label_3/all_images"
    path_arr=[label_1_path,label_2_path,label_3_path]
    if not os.path.isdir(dest_dir_path):
        os.mkdir(dest_dir_path)
    else:
        shutil.rmtree(dest_dir_path)
        os.mkdir(dest_dir_path)
    for i in range(len(path_arr)):
        for j in range(1,num_of_subdir+1):
            sub_dir_path=path_arr[i]+"/{}".format(j)
            for filename in np.sort(os.listdir(sub_dir_path)):
                if filename[0] == '.':
                    continue
                shutil.copy(sub_dir_path+"/"+filename,dest_dir_path)