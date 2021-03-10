import argparse
import os
from src import constants as cons
from src.preprocessing import utils
from src.augmentation import imageaugmentator
import gc
import numpy as np
from shutil import copyfile
import joblib
import random

def main(input_path, output_path, augment_data):

    seed_value = 25

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    utrech_dataset, singapore_dataset, amsterdam_dataset = utils.get_all_images_and_labels(os.path.join(input_path, cons.UTRECHT_FOLDER),
                                                                                           os.path.join(input_path, cons.SINGAPORE_FOLDER),
                                                                                           os.path.join(input_path, cons.AMSTERDAM_FOLDER))

    print(utrech_dataset)
    print(singapore_dataset)
    print(amsterdam_dataset)

    t1_key = "T1_rsfl"
    flair_key = "FLAIR"
    label_key = "wmh"
    mask_key = "T1_bet_mask_rsfl"

    t1_utrecht_paths, flair_utrecht_paths, labels_utrecht_paths, brain_mask_utrecht_paths = utils.get_all_sets_paths(utrech_dataset,
                                                                                             t1_key=t1_key,
                                                                                             flair_key=flair_key,
                                                                                             label_key=label_key,
                                                                                             mask_key=mask_key)
    t1_singapore_paths, flair_singapore_paths, labels_singapore_paths, brain_mask_singapore_paths = utils.get_all_sets_paths(singapore_dataset,
                                                                                             t1_key=t1_key,
                                                                                             flair_key=flair_key,
                                                                                             label_key=label_key,
                                                                                             mask_key=mask_key)
    t1_amsterdam_paths, flair_amsterdam_paths, labels_amsterdam_paths, brain_mask_amsterdam_paths = utils.get_all_sets_paths(amsterdam_dataset,
                                                                                             t1_key=t1_key,
                                                                                             flair_key=flair_key,
                                                                                             label_key=label_key,
                                                                                             mask_key=mask_key)


    print('Utrecht: ', len(t1_utrecht_paths), len(flair_utrecht_paths), len(labels_utrecht_paths))
    print('Singapore: ', len(t1_singapore_paths), len(flair_singapore_paths), len(labels_singapore_paths))
    print('Amsterdam: ', len(t1_amsterdam_paths), len(flair_amsterdam_paths), len(labels_amsterdam_paths))

    # MASKS DATA

    utrecht_masks = utils.get_all_images_np_twod(brain_mask_utrecht_paths)
    singapore_masks = utils.get_all_images_np_twod(brain_mask_singapore_paths)
    amsterdam_masks = utils.get_all_images_np_twod(brain_mask_amsterdam_paths)

    # LABELS DATA

    labels_utrecht, labels_singapore, labels_amsterdam = utils.preprocess_all_labels([labels_utrecht_paths, labels_singapore_paths, labels_amsterdam_paths],
                                                                                     [utrecht_masks, singapore_masks, amsterdam_masks],
                                                                                     cons.SLICE_SHAPE,
                                                                                     cons.REMOVE_TOP_PRCTG,
                                                                                     cons.REMOVE_BOT_PRCTG)

    # T1 DATA

    utrecht_normalized_t1, singapore_normalized_t1, amsterdam_normalized_t1 = utils.preprocess_all_datasets([t1_utrecht_paths, t1_singapore_paths, t1_amsterdam_paths],
                                                                                                            cons.SLICE_SHAPE,
                                                                                                            [utrecht_masks, singapore_masks, amsterdam_masks],
                                                                                                            cons.REMOVE_TOP_PRCTG,
                                                                                                            cons.REMOVE_BOT_PRCTG)

    # FLAIR DATA

    utrecht_stand_flairs, singapore_stand_flairs, amsterdam_stand_flairs = utils.preprocess_all_datasets([flair_utrecht_paths, flair_singapore_paths, flair_amsterdam_paths],
                                                                                                          cons.SLICE_SHAPE,
                                                                                                          [utrecht_masks, singapore_masks, amsterdam_masks],
                                                                                                          cons.REMOVE_TOP_PRCTG, cons.REMOVE_BOT_PRCTG)



    options = np.arange(20)


    for i in range(5):

        # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        # 2. Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed_value)
        # 3. Set `numpy` pseudo-random generator at a fixed value
        np.random.seed(seed_value)

        indices_split = np.random.choice(options, 4, replace=False)
        for idx in indices_split:
            options = np.delete(options, np.where(options == idx))

        print(indices_split, options)
        labels_train, labels_validation, labels_test = utils.full_custom_split([labels_utrecht, labels_singapore, labels_amsterdam],
                                                                               indices_split.copy())

        save_labels_validation, save_labels_test = utils.full_custom_split_for_save([labels_utrecht, labels_singapore, labels_amsterdam],
                                                                               indices_split.copy())


        #DATA CONCAT

        train_t1, validation_t1, test_t1 = utils.full_custom_split([utrecht_normalized_t1, singapore_normalized_t1, amsterdam_normalized_t1],
                                                                                      indices_split.copy())

        save_validation_t1, save_test_t1 = utils.full_custom_split_for_save(
            [utrecht_normalized_t1, singapore_normalized_t1, amsterdam_normalized_t1],
            indices_split.copy())

        train_flair, validation_flair, test_flair = utils.full_custom_split([utrecht_stand_flairs, singapore_stand_flairs, amsterdam_stand_flairs],
                                                                                               indices_split.copy())

        save_validation_flair, save_test_flair = utils.full_custom_split_for_save(
            [utrecht_stand_flairs, singapore_stand_flairs, amsterdam_stand_flairs],
            indices_split.copy())

        save_validation_data = utils.concat_save_data(save_validation_t1, save_validation_flair)
        save_test_data = utils.concat_save_data(save_test_t1, save_test_flair)

        train_data = np.concatenate([train_t1, train_flair], axis=3)

        validation_data = np.concatenate([validation_t1, validation_flair], axis=3)

        test_data = np.concatenate([test_t1, test_flair], axis=3)

        gc.collect()

        # AUGMENTATION
        if augment_data:
            print("Applying data augmentation")
            augmentator = imageaugmentator.ImageAugmentator()
            train_data, labels_train = augmentator.perform_all_augmentations(train_data, labels_train)

        #shuffle
        print(train_data.shape, labels_train.shape)
        datalabels_train, datalabels_validation, datalabels_test = utils.permute_data([(train_data, labels_train),
                                                                                       (validation_data, labels_validation),
                                                                                       (test_data, labels_test)])

        data_train, labels_train = datalabels_train
        data_validation, labels_validation = datalabels_validation
        data_test, labels_test = datalabels_test

        del datalabels_train, datalabels_validation, datalabels_test

        print(i, "TRAIN", data_train.shape, labels_train.shape)
        print(i, "VALIDATION", data_validation.shape, labels_validation.shape)
        print(i, "TEST", data_test.shape, labels_test.shape)

        utils.copy_original_data([np.array(flair_utrecht_paths)[indices_split[0]],
                                  np.array(flair_singapore_paths)[indices_split[0]],
                                  np.array(flair_amsterdam_paths)[indices_split[0]]],
                                 output_path, cons.TEST_FOLDER, i)

        utils.copy_original_data([*np.array(flair_utrecht_paths)[indices_split[1:]],
                                 *np.array(flair_singapore_paths)[indices_split[1:]],
                                 *np.array(flair_amsterdam_paths)[indices_split[1:]]],
                                 output_path, cons.VALIDATION_FOLDER, i)


        masks_test_out_path = os.path.join(output_path, cons.TEST_FOLDER, "raw_copies", f"{i}_masks.pkl")
        masks_test_save = [np.array(utrecht_masks)[indices_split[0]],
                                  np.array(singapore_masks)[indices_split[0]],
                                  np.array(amsterdam_masks)[indices_split[0]]]
        joblib.dump(masks_test_save, masks_test_out_path)

        masks_validation_save = [*np.array(utrecht_masks)[indices_split[1:]],
                                  *np.array(singapore_masks)[indices_split[1:]],
                                  *np.array(amsterdam_masks)[indices_split[1:]]]
        masks_validation_out_path = os.path.join(output_path, cons.VALIDATION_FOLDER, "raw_copies", f"{i}_masks.pkl")
        joblib.dump(masks_validation_save, masks_validation_out_path)


        utils.save_data_labels(os.path.join(output_path, cons.TRAIN_FOLDER), i, data_train, cons.DATA_TRAIN_NAME, labels_train, cons.LABELS_TRAIN_NAME)
        utils.save_data_labels(os.path.join(output_path, cons.VALIDATION_FOLDER), i, data_validation, cons.DATA_VALIDATION_NAME, labels_validation, cons.LABELS_VALIDATION_NAME)
        utils.save_data_labels(os.path.join(output_path, cons.TEST_FOLDER), i, data_test, cons.DATA_TEST_NAME, labels_test, cons.LABELS_TEST_NAME)

        # Saving ordered data
        save_ordered_validation_path = os.path.join(output_path, cons.VALIDATION_FOLDER, cons.ORDERED_DIR)
        if not os.path.exists(save_ordered_validation_path):
            os.mkdir(save_ordered_validation_path)
        utils.save_data_labels(save_ordered_validation_path,
                               i, save_validation_data, cons.DATA_VALIDATION_NAME,
                               save_labels_validation, cons.LABELS_VALIDATION_NAME)


        save_ordered_test_path = os.path.join(output_path, cons.TEST_FOLDER, cons.ORDERED_DIR)
        if not os.path.exists(save_ordered_test_path):
            os.mkdir(save_ordered_test_path)
        utils.save_data_labels(save_ordered_test_path,
                               i, save_test_data, cons.DATA_TEST_NAME,
                               save_labels_test, cons.LABELS_TEST_NAME)




        del data_train, labels_train, data_validation, labels_validation, data_test, labels_test
        gc.collect()


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_path", default="data/raw/noskull_enhanced", type=str, help="Raw data path")
    argparser.add_argument("--output_path", type=str, help="Processed data output path")
    argparser.add_argument("--aug", default=False, type=str2bool, help="Apply or not offline data augmentation.")

    args = argparser.parse_args()

    output_path = os.path.abspath(args.output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    main(os.path.abspath(args.input_path), output_path, args.aug)
