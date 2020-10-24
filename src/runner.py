import numpy as np
from networks.ustepnet import UStepNet
from networks.unet import Unet
from networks.shallowunet import ShallowUnet
from networks.stacknet import StackNet
from networks.resunet import ResUnet
from networks.resunet_custom import CustomResUnet
from networks.attention_unet import AttentionUnet
from preprocessing.imageparser import ImageParser
from augmentation.imageaugmentator import ImageAugmentator
from sklearn.model_selection import train_test_split
from constants import *
import gc
import os
import tensorflow as tf
import utils
import cv2
from evaluation import evaluate

seed_value = 25
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)


parser = ImageParser(path_utrech='../dataset_new_enhanced/Utrecht/subjects',
                     path_singapore='../dataset_new_enhanced/Singapore/subjects',
                     path_amsterdam='../dataset_new_enhanced/GE3T/subjects')
utrech_dataset, singapore_dataset, amsterdam_dataset = parser.get_all_images_and_labels()

print(utrech_dataset)
print(singapore_dataset)
print(amsterdam_dataset)

t1_utrecht, flair_utrecht, labels_utrecht, brain_mask_utrecht = parser.get_all_sets_paths(utrech_dataset)
t1_singapore, flair_singapore, labels_singapore, brain_mask_singapore = parser.get_all_sets_paths(singapore_dataset)
t1_amsterdam, flair_amsterdam, labels_amsterdam, brain_mask_amsterdam = parser.get_all_sets_paths(amsterdam_dataset)

slice_shape = SLICE_SHAPE

print('Utrecht: ', len(t1_utrecht), len(flair_utrecht), len(labels_utrecht))
print('Singapore: ', len(t1_singapore), len(flair_singapore), len(labels_singapore))
print('Amsterdam: ', len(t1_amsterdam), len(flair_amsterdam), len(labels_amsterdam))


"""

MASKS DATA

"""

utrecht_masks = parser.get_all_images_np_twod(brain_mask_utrecht)
singapore_masks = parser.get_all_images_np_twod(brain_mask_singapore)
amsterdam_masks = parser.get_all_images_np_twod(brain_mask_amsterdam)


"""

LABELS DATA

"""

indices_split = np.random.randint(0, 20, 3)

labels_utrecht = parser.preprocess_dataset_labels(labels_utrecht, utrecht_masks, slice_shape,
                                                  REMOVE_TOP_PRCTG, REMOVE_BOT_PRCTG)

labels_singapore = parser.preprocess_dataset_labels(labels_singapore, singapore_masks, slice_shape,
                                                    REMOVE_TOP_PRCTG, REMOVE_BOT_PRCTG)

labels_amsterdam = parser.preprocess_dataset_labels(labels_amsterdam, amsterdam_masks, slice_shape,
                                                    REMOVE_TOP_PRCTG, REMOVE_BOT_PRCTG)

options = set(range(0, 20))

for i in range(5):

    indices_split = np.random.choice(list(options), 4, replace=False)
    for idx in indices_split:
        options.remove(idx)

    utrecht_labels_train, utrecht_labels_test, utrecht_labels_validation = utils.custom_split(labels_utrecht,
                                                                                              indices_split)

    singapore_labels_train, singapore_labels_test, singapore_labels_validation = utils.custom_split(labels_singapore,
                                                                                                    indices_split)

    amsterdam_labels_train, amsterdam_labels_test, amsterdam_labels_validation = utils.custom_split(labels_amsterdam,
                                                                                                    indices_split)

    labels_train = np.concatenate([np.concatenate(utrecht_labels_train),
                                   np.concatenate(singapore_labels_train),
                                   np.concatenate(amsterdam_labels_train)], axis=0)
    labels_train = np.expand_dims(labels_train, axis=-1)
    labels_test = np.concatenate([np.concatenate(utrecht_labels_test),
                                  np.concatenate(singapore_labels_test),
                                  np.concatenate(amsterdam_labels_test)], axis=0)
    labels_test = np.expand_dims(labels_test, axis=-1)
    labels_validation = np.expand_dims(np.concatenate([utrecht_labels_validation,
                                                       singapore_labels_validation,
                                                       amsterdam_labels_validation], axis=0),
                                       axis=-1)




    '''
    
    T1 DATA
    
    '''
    utrecht_normalized_t1 = parser.preprocess_dataset_t1(t1_utrecht, slice_shape, utrecht_masks,
                                                         REMOVE_TOP_PRCTG, REMOVE_BOT_PRCTG)
    singapore_normalized_t1 = parser.preprocess_dataset_t1(t1_singapore, slice_shape, singapore_masks,
                                                           REMOVE_TOP_PRCTG, REMOVE_BOT_PRCTG)
    amsterdam_normalized_t1 = parser.preprocess_dataset_t1(t1_amsterdam, slice_shape, amsterdam_masks,
                                                           REMOVE_TOP_PRCTG, REMOVE_BOT_PRCTG)

    del t1_utrecht, t1_singapore, t1_amsterdam

    '''
    
    FLAIR DATA
    
    '''


    utrecht_stand_flairs = parser.preprocess_dataset_flair(flair_utrecht, slice_shape, utrecht_masks,
                                                           REMOVE_TOP_PRCTG, REMOVE_BOT_PRCTG)
    singapore_stand_flairs = parser.preprocess_dataset_flair(flair_singapore, slice_shape, singapore_masks,
                                                             REMOVE_TOP_PRCTG, REMOVE_BOT_PRCTG)
    amsterdam_stand_flairs = parser.preprocess_dataset_flair(flair_amsterdam, slice_shape, amsterdam_masks,
                                                             REMOVE_TOP_PRCTG, REMOVE_BOT_PRCTG)

    del flair_utrecht, flair_singapore, flair_amsterdam


    '''
    
    DATA CONCAT
    
    '''



    utrecht_t1_train, utrecht_t1_test, utrecht_t1_validation = utils.custom_split(utrecht_normalized_t1,
                                                                         indices_split)

    singapore_t1_train, singapore_t1_test, singapore_t1_validation = utils.custom_split(singapore_normalized_t1,
                                                                               indices_split)

    amsterdam_t1_train, amsterdam_t1_test, amsterdam_t1_validation = utils.custom_split(amsterdam_normalized_t1,
                                                                               indices_split)




    utrecht_flair_train, utrecht_flair_test, utrecht_flair_validation = utils.custom_split(utrecht_stand_flairs,
                                                                         indices_split)

    singapore_flair_train, singapore_flair_test, singapore_flair_validation = utils.custom_split(singapore_stand_flairs,
                                                                               indices_split)

    amsterdam_flair_train, amsterdam_flair_test, amsterdam_flair_validation = utils.custom_split(amsterdam_stand_flairs,
                                                                               indices_split)


    train_t1 = np.concatenate([np.concatenate(utrecht_t1_train),
                               np.concatenate(singapore_t1_train),
                               np.concatenate(amsterdam_t1_train)], axis=0)
    test_t1 = np.concatenate([np.concatenate(utrecht_t1_test),
                              np.concatenate(singapore_t1_test),
                              np.concatenate(amsterdam_t1_test)], axis=0)
    validation_t1 = np.concatenate([utrecht_t1_validation,
                                    singapore_t1_validation,
                                    amsterdam_t1_validation], axis=0)

    train_flair = np.concatenate([np.concatenate(utrecht_flair_train),
                                  np.concatenate(singapore_flair_train),
                                  np.concatenate(amsterdam_flair_train)], axis=0)
    test_flair = np.concatenate([np.concatenate(utrecht_flair_test),
                                 np.concatenate(singapore_flair_test),
                                 np.concatenate(amsterdam_flair_test)], axis=0)
    validation_flair = np.concatenate([utrecht_flair_validation,
                                       singapore_flair_validation,
                                       amsterdam_flair_validation], axis=0)


    train_data = np.concatenate([np.expand_dims(train_t1, axis=-1),
                                 np.expand_dims(train_flair, axis=-1)], axis=3)

    test_data = np.concatenate([np.expand_dims(test_t1, axis=-1),
                                 np.expand_dims(test_flair, axis=-1)], axis=3)

    validation_data = np.concatenate([np.expand_dims(validation_t1, axis=-1),
                                 np.expand_dims(validation_flair, axis=-1)], axis=3)

    gc.collect()
    '''
    
    AUGMENTATION
    
    '''
    print(train_data.shape, labels_train.shape)
    augmentator = ImageAugmentator()
    data_augmented, labels_agumented = augmentator.perform_all_augmentations(train_data, labels_train)

    data_train = np.asanyarray(data_augmented)
    labels_train = np.asanyarray(labels_agumented)
    del data_augmented, labels_agumented


    #data_train, test_data, validation_data = utils.get_coordinates([data_train, test_data, validation_data])


    '''
    
    TRAINING
    
    '''
    idx_swap = np.arange(data_train.shape[0])
    np.random.shuffle(idx_swap)

    data_train = data_train[idx_swap]
    labels_train = labels_train[idx_swap]

    idx_swap = np.arange(test_data.shape[0])
    np.random.shuffle(idx_swap)

    test_data = test_data[idx_swap]
    labels_test = labels_test[idx_swap]

    idx_swap = np.arange(validation_data.shape[0])
    np.random.shuffle(idx_swap)


    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as K
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)


    gc.collect()
    training_name = '20200818_f1_new_weight_v2_{}'.format(i)
    base_path = os.getcwd()

    print(data_train.shape, labels_train.shape, test_data.shape, labels_test.shape)

    unet = AttentionUnet(img_shape=data_train.shape[1:])
    unet.train(data_train, labels_train, (test_data, labels_test), training_name, base_path, epochs=40, batch_size=15)


    '''
    
    VALIDATING
    
    '''

    predictions = unet.predict_and_save(validation_data, labels_validation)
    #evaluation = evaluate.evaluarSR(np.squeeze(predictions), labels_validation)

    #print(evaluation)
    """
    
    VISUALIZING
    
    """
    del data_train, labels_train, training_name
    gc.collect()
    #unet.save_visualize_activations(validation_data, labels_validation, batch_size=30)
