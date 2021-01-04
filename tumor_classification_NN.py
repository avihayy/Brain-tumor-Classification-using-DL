"""
tumor_classification_NN.py contains all the functions that build and operating on all our NN: segmentation and classification.
include:
1) function for building the segmentation and classification NN (unet_model(),create_classification_model())
2) functions for training the segmentation and classification NN (train_Unet_segmentation_model(),train_all_classification_sub_models(),train_classification_model())
3) functions for testing the segmentation and classification NN and the complete network (test_segmentation_model(),test_specific_model(),test_model()).
"""


from PIL import Image
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    Dropout,
    SpatialDropout2D,
    Input,
    concatenate,
    multiply,
    add,
    Activation,)
from prepare_dataset import *
import time



class UnetPets(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size +(1,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            f = h5py.File(path, 'r')
            data = f.get('cjdata')
            image = data.get('image')
            img_data = image.value
            img_data_resize = cv2.resize(img_data, self.img_size)
            img_data_resize = img_data_resize / np.max(img_data_resize)
            img = Image.fromarray(img_data_resize)
            x[j] = np.expand_dims(img, 2)
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            f = h5py.File(path, 'r')
            data = f.get('cjdata')
            mask = data.get('tumorMask')
            mask_data = mask.value
            img = Image.fromarray(cv2.resize(mask_data, self.img_size),"L")
            y[j] = np.expand_dims(img, 2)
        return x, y



def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def attention_gate(inp_1, inp_2, n_intermediate_filters):
    """Attention gate. Compresses both inputs to n_intermediate_filters filters before processing.
       Implemented as proposed by Oktay et al. in their Attention U-net, see: https://arxiv.org/abs/1804.03999.
    """
    inp_1_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_1)
    inp_2_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_2)

    f = Activation("relu")(add([inp_1_conv, inp_2_conv]))
    g = Conv2D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(f)
    h = Activation("sigmoid")(g)
    return multiply([inp_1, h])


def attention_concat(conv_below, skip_connection):
    """Performs concatenation of upsampled conv_below with attention gated version of skip-connection
    """
    below_filters = conv_below.get_shape().as_list()[-1]
    attention_across = attention_gate(skip_connection, conv_below, below_filters)
    return concatenate([conv_below, attention_across])


def conv2d_block(
    inputs,
    use_batch_norm=True,
    dropout=0.3,
    dropout_type="spatial",
    filters=16,
    kernel_size=(3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
):

    if dropout_type == "spatial":
        DO = SpatialDropout2D
    elif dropout_type == "standard":
        DO = Dropout
    else:
        raise ValueError(
            f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
        )

    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = DO(dropout)(c)
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def unet_model(
    input_shape,
    num_classes=1,
    activation="relu",
    use_batch_norm=True,
    dropout=0.3,
    dropout_change_per_layer=0.0,
    dropout_type="spatial",
    use_dropout_on_upsampling=False,
    use_attention=False,
    filters=16,
    num_layers=4,
    output_activation="sigmoid",
):

    """
    credits to:https://github.com/karolzak/keras-unet
    Arguments:
    1)input_shape: 3D Tensor of shape (x, y, num_channels)
    2)num_classes (int): Unique classes in the output mask. Should be set to 1 for binary segmentation
    3)activation (str): A keras.activations.Activation to use. ReLu by default.
    4)use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers
    5)upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part
    6)dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off
    7)dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block
    8)dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]
    9)use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network
    10)use_attention (bool): Whether to use an attention dynamic when concatenating with the skip-connection, implemented as proposed by Oktay et al. [3]
    11)filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block
    12)num_layers (int): Number of total layers in the encoder not including the bottleneck layer
    13)output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation

    Returns:
    1)model (keras.models.Model): The built U-Net

    Raises:
    ValueError: If dropout_type is not one of "spatial" or "standard"
    [1]: https://arxiv.org/abs/1505.04597
    [2]: https://arxiv.org/pdf/1411.4280.pdf
    [3]: https://arxiv.org/abs/1804.03999
    """


    upsample = upsample_conv


    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    x = conv2d_block(
        inputs=x,
        filters=filters,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        activation=activation,
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        if use_attention:
            x = attention_concat(conv_below=x, skip_connection=conv)
        else:
            x = concatenate([x, conv])
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model


def train_Unet_segmentation_model(path_for_save,how_to_separate,epochs = 15,batch_size=32,img_size=(256,256)):
    input_dir_train = "tumor_dataset/train_set/train_" + how_to_separate + "_separation/train_dataset_for_unet_segmentation_model/"
    input_dir_val = "tumor_dataset/validation_set/validation_"+how_to_separate+"_separation/all_labels/"
    input_img_paths_train = sorted([
        os.path.join(input_dir_train, fname)
        for fname in os.listdir(input_dir_train)
        if fname.endswith(".mat") and not fname.startswith(".")

    ])
    input_img_paths_val = sorted([
        os.path.join(input_dir_val, fname)
        for fname in os.listdir(input_dir_val)
        if fname.endswith(".mat") and not fname.startswith(".")

    ])

    target_path_train = input_img_paths_train
    target_path_val = input_img_paths_val
    # Free up RAM
    keras.backend.clear_session()

    # Build model
    model = unet_model(img_size + (1,), use_attention=True)
    model.summary()

    # Instantiate data Sequences for each split
    train_gen = UnetPets(
        batch_size, img_size, input_img_paths_train, target_path_train
    )
    val_gen = UnetPets(batch_size, img_size, input_img_paths_val, target_path_val)

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])
    callbacks = [keras.callbacks.ModelCheckpoint(path_for_save+".h5", save_best_only=True)]
    model.fit(train_gen, epochs=epochs, callbacks=callbacks,validation_data=val_gen)
    model.save(path_for_save, save_format='tf')


def create_classification_model(img_size,num_classes,with_mask):

    vgg16_model_img = VGG16(include_top=False, input_shape=img_size + (3,))
    vgg16_model_mask = VGG16(include_top=False, input_shape=img_size + (3,))

    for layer in vgg16_model_mask.layers:
        layer._name = layer.name + str('_M')

    m_1 = layers.Conv2D(512, 3, padding="valid",activation="relu")(vgg16_model_mask.layers[-1].output)
    m_1 = layers.Flatten()(m_1)
    m_1 = layers.BatchNormalization()(m_1)
    m_1 = layers.Dropout(0.25)(m_1)
    m_1 = layers.Dense(128, activation="relu")(m_1)
    m_1 = layers.BatchNormalization()(m_1)

    x_1 = layers.Conv2D(512, 3, padding="valid",activation="relu")(vgg16_model_img.layers[-1].output)
    x_1 = layers.Flatten()(x_1)
    x_1 = layers.BatchNormalization()(x_1)
    x_1 = layers.Dropout(0.25)(x_1)
    x_1 = layers.Dense(128, activation="relu")(x_1)
    x_1 = layers.BatchNormalization()(x_1)

    if with_mask == True:
         concatenate=layers.concatenate([x_1,m_1])
         x_1 = layers.Dropout(0.25)(concatenate)
    else:
         x_1 = layers.Dropout(0.25)(x_1)

    x_1 = layers.Dense(64, activation="relu")(x_1)
    x_1 = layers.BatchNormalization()(x_1)
    x_1 = layers.Dropout(0.25)(x_1)
    x_1 = layers.Dense(32, activation="relu")(x_1)
    x_1 = layers.Dense(num_classes)(x_1)
    x_1 = layers.Softmax()(x_1)
    output = x_1
    if with_mask == True:
        model = keras.Model([vgg16_model_img.input,vgg16_model_mask.input], output)
    else:
        model = keras.Model(vgg16_model_img.input, output)
    return model


def train_classification_model(model,x_train,y_train,x_val,y_val,with_mask,epochs=10,batch_size=32,save_model=True,path_for_save="saved_classifier_model/MyModel"):
    if with_mask==True:
        num_of_layers_to_freeze = 36
    else:
        num_of_layers_to_freeze = 18
    for i in range(num_of_layers_to_freeze):
        model.layers[i].trainable = False
        print("freeze layer: %s" %  model.layers[i].name)
    model.compile(optimizer="Adam", loss="SparseCategoricalCrossentropy", metrics=['accuracy'])
    model.summary()

    callbacks = [keras.callbacks.ModelCheckpoint(path_for_save+".h5", save_best_only=True)]
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_val,y_val),callbacks=callbacks)
    if save_model:
        model.save_weights(path_for_save+"_final.h5", save_format='tf')
        print("save model at:%s" % path_for_save)


def train_all_classification_sub_models(epochs_classification_model,batch_size_classification_model,img_size,num_classes,with_mask,model_save_path,split_dataset_to_N,itr,how_to_separate):
    data_type_arr = ["label_1_2", "label_1_3", "label_2_3"]
    for data_type in data_type_arr:
        keras.backend.clear_session()
        model = create_classification_model(img_size, num_classes, with_mask)
        x_val,x_mask_val, y_val = prepare_data_for_validation(img_size, data_type,how_to_separate)
        model_save_fall_path = model_save_path + data_type
        print("train model type=%s" % data_type)
        for i in range(itr):
            print("######################################################################################")
            print("start iteration %s/%s" % (i + 1, itr))
            curr_dataset = (i % split_dataset_to_N) + 1
            print("train dataset #%s" % curr_dataset)
            x_train, x_train_mask, y_train = prepare_data_for_train(data_type, img_size, curr_dataset,how_to_separate)

            if with_mask == True:
                train_classification_model(model, [x_train, x_train_mask], y_train, [x_val, x_mask_val], y_val,
                                           with_mask, epochs_classification_model, batch_size_classification_model,
                                           save_model=True, path_for_save=model_save_fall_path)
            else:
                train_classification_model(model, x_train, y_train, x_val, y_val, with_mask,
                                           epochs_classification_model, batch_size_classification_model,
                                           save_model=True, path_for_save=model_save_fall_path)
            print("######################################################################################")

        test_specific_model(model,img_size,data_type,with_mask)
        model.load_weights(model_save_fall_path+".h5")
        test_specific_model(model, img_size, data_type, with_mask)


def test_specific_model(model,img_size,model_data_type,with_mask):
    x_test,x_mask_tetst, y_test = prepare_data_for_test(img_size, model_data_type)
    if with_mask==True:
        pred_label = np.argmax(model.predict([x_test,x_mask_tetst]), axis=1)
    else:
        pred_label = np.argmax(model.predict(x_test), axis=1)
    true_label = y_test
    test_accuracy = np.sum(pred_label == true_label)/len(pred_label)
    print("test accuracy for model %s is: %s ,num of test images=%s" % (model_data_type, test_accuracy, len(pred_label)))


def dice_coef(y_true, y_pred, smooth=1):

    intersection = K.sum(K.abs(y_true * y_pred)).numpy()
    dice = (2 * intersection + smooth) / (K.sum(K.abs(y_true)).numpy() + K.sum(K.abs(y_pred)).numpy() + smooth)
    return dice


def test_segmentation_model(segmentation_model,img_size):
    model_data_type = "all_labels"
    x_test, x_mask_GT, y_test = prepare_data_for_test(img_size, model_data_type)
    x_mask_pred = segmentation_model.predict(x_test[:, :, :, 0])
    x_mask_pred = x_mask_pred.round()
    sum_of_dice_coefficient=0
    for i in range(x_mask_pred.shape[0]):
        sum_of_dice_coefficient = sum_of_dice_coefficient+dice_coef(x_mask_GT[i,:,:,0],x_mask_pred[i,:,:,0],smooth=0)
    dice_coefficient = sum_of_dice_coefficient/x_mask_pred.shape[0]
    print("dice coefficient = %.2f" % dice_coefficient)

def test_model(segmentation_model,model_label_1_2,model_label_1_3,model_label_2_3,img_size,with_mask,how_to_separate,use_pred_mask):
    model_data_type="all_labels"
    x_test,x_mask_GT, y_test = prepare_data_for_test(img_size, model_data_type,how_to_separate)
    start_time=time.time()
    x_mask_pred = segmentation_model.predict(x_test[:,:,:,0])
    x_mask_1d=x_mask_pred.round()
    x_mask_3d_pred = np.concatenate((x_mask_1d, x_mask_1d, x_mask_1d), axis=3)
    if with_mask==True:
        if not use_pred_mask:
            pred_model_label_1_2 = model_label_1_2.predict([x_test,x_mask_GT])
            pred_model_label_1_3 = model_label_1_3.predict([x_test,x_mask_GT])
            pred_model_label_2_3 = model_label_2_3.predict([x_test,x_mask_GT])
        else:
            pred_model_label_1_2 = model_label_1_2.predict([x_test, x_mask_3d_pred])
            pred_model_label_1_3 = model_label_1_3.predict([x_test, x_mask_3d_pred])
            pred_model_label_2_3 = model_label_2_3.predict([x_test, x_mask_3d_pred])

    else:
        pred_model_label_1_2 = model_label_1_2.predict(x_test)
        pred_model_label_1_3 = model_label_1_3.predict(x_test)
        pred_model_label_2_3 = model_label_2_3.predict(x_test)

    prob_to_be_label_1 = (pred_model_label_1_2[:,0] + pred_model_label_1_3[:,0])
    prob_to_be_label_2 = (pred_model_label_1_2[:,1] + pred_model_label_2_3[:,0])
    prob_to_be_label_3 = (pred_model_label_1_3[:, 1] + pred_model_label_2_3[:, 1])
    prob_to_be_label_1_column = np.reshape(prob_to_be_label_1,(-1,1))
    prob_to_be_label_2_column = np.reshape(prob_to_be_label_2,(-1,1))
    prob_to_be_label_3_column = np.reshape(prob_to_be_label_3,(-1,1))
    pred_model_soft=np.concatenate((prob_to_be_label_1_column,prob_to_be_label_2_column,prob_to_be_label_3_column),axis=1)
    pred_model_hard=np.argmax(pred_model_soft,axis=1)
    true_label = y_test
    end_time = time.time()
    print_results(true_label, pred_model_hard)
    print("total prediction time= {:.3f}sec, time per image= {:.3f}sec".format(end_time-start_time,(end_time-start_time)/len(true_label)))
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


def print_results(true_label,pred_model):
    num_of_Meningioma_class_1 = sum(true_label == 0)
    num_of_Glioma_class_2 = sum(true_label == 1)
    num_of_Pituitary_class_3 = sum(true_label == 2)

    Meningioma_class_1_idx = np.where(true_label == 0)
    Glioma_class_2_idx = np.where(true_label == 1)
    Pituitary_class_3_idx = np.where(true_label == 2)

    num_of_Meningioma_class_1_pred_as_Meningioma_class_1 = sum(pred_model[Meningioma_class_1_idx] == 0)
    num_of_Meningioma_class_1_pred_as_Glioma_class_2=sum(pred_model[Meningioma_class_1_idx]==1)
    num_of_Meningioma_class_1_pred_as_Pituitary_class_3 = sum(pred_model[Meningioma_class_1_idx] == 2)

    num_of_Glioma_class_2_pred_as_Meningioma_class_1 = sum(pred_model[Glioma_class_2_idx] == 0)
    num_of_Glioma_class_2_pred_as_Glioma_class_2 = sum(pred_model[Glioma_class_2_idx] == 1)
    num_of_Glioma_class_2_pred_as_Pituitary_class_3 = sum(pred_model[Glioma_class_2_idx] == 2)

    num_of_Pituitary_class_3_pred_as_Meningioma_class_1 = sum(pred_model[Pituitary_class_3_idx] == 0)
    num_of_Pituitary_class_3_pred_as_Glioma_class_2 = sum(pred_model[Pituitary_class_3_idx] == 1)
    num_of_Pituitary_class_3_pred_as_Pituitary_class_3 = sum(pred_model[Pituitary_class_3_idx] == 2)
    total_accuracy = (num_of_Meningioma_class_1_pred_as_Meningioma_class_1+num_of_Glioma_class_2_pred_as_Glioma_class_2+num_of_Pituitary_class_3_pred_as_Pituitary_class_3)/(num_of_Meningioma_class_1+num_of_Glioma_class_2+num_of_Pituitary_class_3)*100

    percision_Meningioma_class_1= num_of_Meningioma_class_1_pred_as_Meningioma_class_1/(num_of_Meningioma_class_1_pred_as_Meningioma_class_1+num_of_Glioma_class_2_pred_as_Meningioma_class_1+num_of_Pituitary_class_3_pred_as_Meningioma_class_1)*100
    percision_Glioma_class_2    = num_of_Glioma_class_2_pred_as_Glioma_class_2 / (num_of_Glioma_class_2_pred_as_Glioma_class_2+num_of_Meningioma_class_1_pred_as_Glioma_class_2+num_of_Pituitary_class_3_pred_as_Glioma_class_2) * 100
    percision_Pituitary_class_3 = num_of_Pituitary_class_3_pred_as_Pituitary_class_3 / (num_of_Pituitary_class_3_pred_as_Pituitary_class_3+num_of_Glioma_class_2_pred_as_Pituitary_class_3+num_of_Meningioma_class_1_pred_as_Pituitary_class_3) * 100

    recall_Meningioma_class_1 = num_of_Meningioma_class_1_pred_as_Meningioma_class_1/num_of_Meningioma_class_1*100
    recall_Glioma_class_2 = num_of_Glioma_class_2_pred_as_Glioma_class_2/num_of_Glioma_class_2*100
    recall_Pituitary_class_3 = num_of_Pituitary_class_3_pred_as_Pituitary_class_3/num_of_Pituitary_class_3*100

    average_precision = np.mean([percision_Meningioma_class_1,percision_Glioma_class_2,percision_Pituitary_class_3])
    average_recall = np.mean([recall_Meningioma_class_1,recall_Glioma_class_2,recall_Pituitary_class_3])
    F1_score = (2*average_precision*average_recall)/(average_precision+average_recall)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%results%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Number of tested images=%s" % len(true_label))
    print("_____________________________________________________________________________________________________________________________________________")
    print("|                          |     Class 1 (Meningioma)   |        Class 2 (Glioma)        |   Class 3 (Pituitary)    |       recall         |")
    print("|   Class 1 (Meningioma)   |           %3d              |           %3d                  |      %3d                 |       %3.1f%%          |" % (num_of_Meningioma_class_1_pred_as_Meningioma_class_1,num_of_Meningioma_class_1_pred_as_Glioma_class_2,num_of_Meningioma_class_1_pred_as_Pituitary_class_3,recall_Meningioma_class_1))
    print("|   Class 2 (Glioma)       |           %3d              |           %3d                  |      %3d                 |       %3.1f%%          |" % (num_of_Glioma_class_2_pred_as_Meningioma_class_1,num_of_Glioma_class_2_pred_as_Glioma_class_2,num_of_Glioma_class_2_pred_as_Pituitary_class_3,recall_Glioma_class_2))
    print("|   Class 3 (Pituitary)    |           %3d              |           %3d                  |      %3d                 |       %3.1f%%          |" % (num_of_Pituitary_class_3_pred_as_Meningioma_class_1,num_of_Pituitary_class_3_pred_as_Glioma_class_2,num_of_Pituitary_class_3_pred_as_Pituitary_class_3,recall_Pituitary_class_3))
    print("|   precision              |           %3.1f%%            |           %3.1f%%                |      %3.1f%%               |       %3.1f%%          |" % (percision_Meningioma_class_1,percision_Glioma_class_2,percision_Pituitary_class_3,total_accuracy))
    print("_____________________________________________________________________________________________________________________________________________")
    print("average precision = %3.1f%%" % average_precision)
    print("average recall = %3.1f%%" % average_recall)
    print("F1 score = %3.1f%%" % F1_score)
    print("total accuracy = %3.1f%%" % total_accuracy)



