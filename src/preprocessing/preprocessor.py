import argparse
import os
from src import constants as cons
from src.preprocessing import utils
from src.augmentation import imageaugmentator
import gc
import numpy as np

def main(input_path, output_path):

    utrech_dataset, singapore_dataset, amsterdam_dataset = utils.get_all_images_and_labels(os.path.join(input_path, cons.UTRECHT_FOLDER),
                                                                                           os.path.join(input_path, cons.SINGAPORE_FOLDER),
                                                                                           os.path.join(input_path, cons.AMSTERDAM_FOLDER))

    print(utrech_dataset)
    print(singapore_dataset)
    print(amsterdam_dataset)

    t1_utrecht, flair_utrecht, labels_utrecht, brain_mask_utrecht = utils.get_all_sets_paths(utrech_dataset)
    t1_singapore, flair_singapore, labels_singapore, brain_mask_singapore = utils.get_all_sets_paths(singapore_dataset)
    t1_amsterdam, flair_amsterdam, labels_amsterdam, brain_mask_amsterdam = utils.get_all_sets_paths(amsterdam_dataset)


    print('Utrecht: ', len(t1_utrecht), len(flair_utrecht), len(labels_utrecht))
    print('Singapore: ', len(t1_singapore), len(flair_singapore), len(labels_singapore))
    print('Amsterdam: ', len(t1_amsterdam), len(flair_amsterdam), len(labels_amsterdam))

    # MASKS DATA

    utrecht_masks = utils.get_all_images_np_twod(brain_mask_utrecht)
    singapore_masks = utils.get_all_images_np_twod(brain_mask_singapore)
    amsterdam_masks = utils.get_all_images_np_twod(brain_mask_amsterdam)

    # LABELS DATA

    indices_split = np.random.randint(0, 20, 3)

    labels_utrecht, labels_singapore, labels_amsterdam = utils.preprocess_all_labels([labels_utrecht, labels_singapore, labels_amsterdam],
                                                                                     [utrecht_masks, singapore_masks, amsterdam_masks],
                                                                                     cons.SLICE_SHAPE,
                                                                                     cons.REMOVE_TOP_PRCTG,
                                                                                     cons.REMOVE_BOT_PRCTG)

    # T1 DATA

    utrecht_normalized_t1, singapore_normalized_t1, amsterdam_normalized_t1 = utils.preprocess_all_datasets([t1_utrecht, t1_singapore, t1_amsterdam],
                                                                                                            cons.SLICE_SHAPE,
                                                                                                            [utrecht_masks, singapore_masks, amsterdam_masks],
                                                                                                            cons.REMOVE_TOP_PRCTG,
                                                                                                            cons.REMOVE_BOT_PRCTG)

    # FLAIR DATA

    utrecht_stand_flairs, singapore_stand_flairs, amsterdam_stand_flairs = utils.preprocess_all_datasets([flair_utrecht, flair_singapore, flair_amsterdam],
                                                                                                          cons.SLICE_SHAPE,
                                                                                                          [utrecht_masks, singapore_masks, amsterdam_masks],
                                                                                                          cons.REMOVE_TOP_PRCTG, cons.REMOVE_BOT_PRCTG)



    options = set(range(0, 20))


    for i in range(5):


        indices_split = np.random.choice(list(options), 4, replace=False)
        for idx in indices_split:
            options.remove(idx)


        labels_train, labels_validation, labels_test = utils.full_custom_split([labels_utrecht, labels_singapore, labels_amsterdam],
                                                                               indices_split)


        #DATA CONCAT

        train_t1, validation_t1, test_t1 = utils.full_custom_split([utrecht_normalized_t1, singapore_normalized_t1, amsterdam_normalized_t1],
                                                                                      indices_split)

        train_flair, validation_flair, test_flair = utils.full_custom_split([utrecht_stand_flairs, singapore_stand_flairs, amsterdam_stand_flairs],
                                                                                               indices_split)


        train_data = np.concatenate([train_t1, train_flair], axis=3)

        validation_data = np.concatenate([validation_t1, validation_flair], axis=3)

        test_data = np.concatenate([test_t1, test_flair], axis=3)

        gc.collect()

        # AUGMENTATION

        print(train_data.shape, labels_train.shape)
        augmentator = imageaugmentator.ImageAugmentator()
        data_augmented, labels_agumented = augmentator.perform_all_augmentations(train_data, labels_train)

        #shuffle

        datalabels_train, datalabels_validation, datalabels_test = utils.permute_data([(data_augmented, labels_agumented),
                                                                                       (validation_data, labels_validation),
                                                                                       (test_data, labels_test)])

        data_train, labels_train = datalabels_train
        data_validation, labels_validation = datalabels_validation
        data_test, labels_test = datalabels_test

        print("TRAIN", data_train.shape, labels_train.shape)
        print("VALIDATION", data_validation.shape, labels_validation.shape)
        print("TEST", data_test.shape, labels_test.shape)

        utils.save_data_labels(os.path.join(output_path, i, cons.TRAIN_FOLDER), data_train, cons.DATA_TRAIN_NAME, labels_train, cons.LABELS_TRAIN_NAME)
        utils.save_data_labels(os.path.join(output_path, i, cons.VALIDATION_FOLDER), data_validation, cons.DATA_VALIDATION_NAME, labels_validation, cons.LABELS_VALIDATION_NAME)
        utils.save_data_labels(os.path.join(output_path, i, cons.TEST_FOLDER), data_test, cons.DATA_TEST_NAME, labels_test, cons.LABELS_TEST_NAME)



if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_path", default="data/raw/noskull_enhanced", help="Raw data path")
    argparser.add_argument("--output_path", help="Processed data output path")

    args = argparser.parse_args()

    output_path = os.path.abspath(args.output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    main(os.path.abspath(args.input_path), output_path)
