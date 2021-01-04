# Brain-tumor-Classification-using-DL
We present here a new CNN architecture for brain tumor classification of three tumor types (which mentioned above), our CNN combined tumor images and tumor masks for classifying brain tumors in T1-weighted contrast-enhanced MRI (magnetic resonance images) images.
In order to preform classification to the three types of the brain tumors we mentioned above we used 2 networks. The first one is a segmentation network which create a binary mask of the tumor based on Unet. The second one is a classification network which based on transfer learning using VGG16 network. This network received 2 inputs- the image data and it’s corresponding mask which we get from the segmentation network.  
In order to receive more accurate classification, our network contains 3 sub-networks and each network is performing a classification between 2 tumors types. The final classification result is achieved by the highest probability score.
The experiment is conducted on a dataset of 3064 images which contain three types of brain tumor (glioma, meningioma, pituitary). We achieved, using our CNN model, a high testing accuracy of 97.8% on random approach an 97.2% on subject approach, average precision of 97.5% on random approach and 97.3% on subject approach and an average recall of 97.6% on random approach and 96.4% on subject approach. The proposed system exhibited satisfying accuracy on the dataset and outperformed many of the prominent existing methods.

random approach- according to this approach we randomly divide the data into 3 parts.

subject approach - according to this approach we split the data into 3 parts according to patients where the data from a single subject could only be found in one of the sets. Each set, therefore, contained data from a couple of subjects regardless of the tumor class, referred to as subject approach.
Note- In both approaches we divided the data into 3 parts which includes train, validation and test.

This repository contains the following files:
1. prepere_dataset.py contains all the functions needed for prepare the dataset for train validation and test, include: 
    1) separating the dataset into train validation and test according to random approach or subject(patient) approach.
    2) reading the images and there masks from .mat files and resizing them to 256x256.
    
2. tumor_classification_NN.py contains all the functions that build and operating on all our NN: segmentation and classification.
include:
    1) function for building the segmentation and classification NN (unet_model(),create_classification_model())
    2) functions for training the segmentation and classification NN (train_Unet_segmentation_model(),train_all_classification_sub_models(),train_classification_model())
    3) functions for testing the segmentation and classification NN and the complete network (test_segmentation_model(),test_specific_model(),test_model()).

3. main file- project_run.ipynb - for running the project in Google Colab. 

Note- the link for Google Colad is attached , the readme for google colab is also located there.

**link for Google Colab** - https://drive.google.com/drive/folders/1u-reMvye_VSz3PEW6LB27JOc4dTityQt?usp=sharing
**link for the dataset** -  https://drive.google.com/drive/folders/1hZnBShvi4b8hFQIFrM4GOrIb6cLIgDAj
