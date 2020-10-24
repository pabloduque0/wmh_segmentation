import os
import gzip
import shutil
import cv2
import numpy as np
import subprocess
from scipy.stats import norm
from constants import *
import matplotlib.pyplot as plt
import SimpleITK

class ImageParser():

    def __init__(self, path_utrech='../Utrecht/subjects',
                 path_singapore='../Singapore/subjects',
                 path_amsterdam='../GE3T/subjects'):
        self.path_utrech = path_utrech
        self.path_singapore = path_singapore
        self.path_amsterdam = path_amsterdam


    def get_all_image_paths(self):
        paths = []

        for root, dirs, files in os.walk('../'):
            for file in files:
                filepath = root + '/' + file

                if file.endswith('.gz') and file[:-3] not in files:
                    with gzip.open(filepath, 'rb') as f_in:
                        with open(filepath[:-3], 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                if file.startswith('brain') and file.endswith('.nii'):
                    paths.append(filepath)

        return paths


    def get_all_images_and_labels(self):
        utrech_dataset = self.get_images_and_labels(self.path_utrech)
        singapore_dataset = self.get_images_and_labels(self.path_singapore)
        amsterdam_dataset = self.get_images_and_labels(self.path_amsterdam)

        return utrech_dataset, singapore_dataset, amsterdam_dataset


    def get_images_and_labels(self, path):
        full_dataset = []
        data_and_labels = {}
        package_limit = 8
        for root, dirs, files in os.walk(path):
            for file in files:
                filepath = os.path.join(root, file)
                key = self.get_key(file)
                if file == 'wmh.nii.gz':
                    data_and_labels[key] = filepath

                length = len(data_and_labels)
                if '/pre/' in filepath and self.is_file_desired(file) and length < package_limit and length > 0:
                    data_and_labels[key] = filepath
                    if len(data_and_labels) == package_limit:
                        full_dataset.append(data_and_labels.copy())
                        print(data_and_labels)
                        data_and_labels.clear()

        return full_dataset

    def get_all_sets_paths(self, dataset_paths):

        t1 = [row["t1_coreg_brain"] for row in dataset_paths]
        flair = [row["new_flair_enhanced"] for row in dataset_paths]
        labels = [row["label"] for row in dataset_paths]
        common_mask = [row["common_mask"] for row in dataset_paths]

        return t1, flair, labels, common_mask



    def preprocess_dataset_t1(self, data_t1, slice_shape, masks, remove_pct_top, remove_pct_bot):

        data_t1 = self.get_all_images_np_twod(data_t1)
        data_t1 = np.asanyarray(data_t1) * np.asanyarray(masks)
        resized_t1 = self.resize_slices(data_t1, slice_shape)
        resized_t1 = self.remove_top_bot_slices(resized_t1, masks, remove_pct_top, remove_pct_bot)
        stand_t1 = self.standarize(resized_t1)
        normalized_t1 = self.normalize_minmax(stand_t1)
        return normalized_t1

    def preprocess_dataset_flair(self, data_flair, slice_shape, masks, remove_pct_top, remove_pct_bot):

        data_flair = self.get_all_images_np_twod(data_flair)
        data_flair = np.asanyarray(data_flair) * np.asanyarray(masks)
        resized_flairs = self.resize_slices(data_flair, slice_shape)
        resized_flairs = self.remove_top_bot_slices(resized_flairs, masks, remove_pct_top, remove_pct_bot)
        stand_flairs = self.standarize(resized_flairs)
        norm_flairs = self.normalize_minmax(stand_flairs)

        return norm_flairs

    def preprocess_dataset_labels(self, label_paths, masks, slice_shape, remove_pct_top, remove_pct_bot):

        labels_imgs = self.get_all_images_np_twod(label_paths)
        labels_resized = self.resize_slices(labels_imgs, slice_shape)
        labels_resized = self.remove_third_label(labels_resized)
        """
        for label, mask in zip(labels_resized, masks):
            remove_bot = int(np.ceil(label.shape[0] * remove_pct_bot))
            remove_top = int(np.ceil(label.shape[0] * remove_pct_top))
            first_slice_brain, last_slice_brain = self.get_first_last_slice(label)
            first_slice_mask, last_slice_mask = self.get_first_last_slice(mask)
            print("Label: ", label.shape, first_slice_brain, last_slice_brain)
            print("Mask: ", mask.shape, first_slice_mask + remove_bot, last_slice_mask - remove_top + 1)
            print("\n")
        """
        labels_resized = self.remove_top_bot_slices(labels_resized, masks, remove_pct_top, remove_pct_bot)

        return labels_resized


    def is_file_desired(self, file_name):
        possibilities = {"FLAIR_masked.nii.gz",
                            "FLAIR.nii.gz",
                            "FLAIR_bet.nii.gz",
                            "T1_masked.nii.gz",
                            "T1.nii.gz",
                            "T1_bet.nii.gz",
                            "T1_bet_mask.nii.gz",
                            "FLAIR_enhanced_lb_masked.nii.gz",
                            "FLAIR_enhanced_lb.nii.gz",
                            "FLAIR-enhanced.nii.gz",
                            "T1_bet_mask_rsfl.nii.gz",
                            "T1_rsfl.nii.gz"}
        return file_name in possibilities

    def get_key(self, file_name):

        possibilities = {"FLAIR_masked.nii.gz": "flair_masked",
                         "FLAIR.nii.gz": "flair",
                         "FLAIR_bet.nii.gz": "flair_bet",
                         "T1_masked.nii.gz": "t1_masked",
                         "T1.nii.gz": "t1",
                         "T1_bet.nii.gz": "t1_bet",
                         "T1_bet_mask.nii.gz": "new_mask",
                         "wmh.nii.gz": "label",
                         "FLAIR_enhanced_lb_masked.nii.gz": "enhanced_masked",
                         "FLAIR_enhanced_lb.nii.gz": "enhanced",
                         "FLAIR-enhanced.nii.gz": "new_flair_enhanced",
                         "T1_bet_mask_rsfl.nii.gz": "common_mask",
                         "T1_rsfl.nii.gz": "t1_coreg_brain"}

        if file_name not in possibilities:
            return None

        return possibilities[file_name]


    def remove_top_bot_slices(self, dataset, masks, remove_pct_top, remove_pct_bot):

        output_images = []
        for image, mask in zip(dataset, masks):
            remove_bot = int(np.ceil(image.shape[0] * remove_pct_bot))
            remove_top = int(np.ceil(image.shape[0] * remove_pct_top))
            first_slice_brain, last_slice_brain = self.get_first_last_slice(mask)
            output_img = image[first_slice_brain + remove_bot: last_slice_brain - remove_top + 1]
            output_images.append(output_img)

        return output_images


    def get_all_images_np_twod(self, paths_list, wrong_shape=(232, 256)):

        patient_list = []
        for path in paths_list:
            image = SimpleITK.ReadImage(path)
            np_image = SimpleITK.GetArrayFromImage(image)

            if np_image.shape[1:] == wrong_shape: #ad hoc fix for Singapore
                # only for even differences
                diff = (wrong_shape[1] - wrong_shape[0]) // 2
                np_image = np.pad(np_image, [(0, 0), (diff, diff), (0, 0)], 'constant', constant_values=0)
                np_image = np_image[..., diff:-diff]
                print('Corrected axises: ', path,  np_image.shape)

            patient_list.append(np_image)

        return patient_list

    def resize_slices(self, image_list, to_slice_shape):

        resized_list = []

        for image in image_list:
            resized_image = []
            for slice in image:
                slice_copy = slice.copy()

                if slice.shape[0] < to_slice_shape[0]:
                    diff = to_slice_shape[0] - slice.shape[0]
                    if self.is_odd(diff):
                        slice_copy = cv2.copyMakeBorder(slice_copy, diff//2, diff//2 + 1, 0, 0,
                                                        cv2.BORDER_CONSTANT,
                                                        value=0.0)
                    else:
                        slice_copy = cv2.copyMakeBorder(slice_copy, diff // 2, diff // 2, 0, 0,
                                                        cv2.BORDER_CONSTANT,
                                                        value=0.0)

                elif slice.shape[0] > to_slice_shape[0]:
                    diff = slice.shape[0] - to_slice_shape[0]
                    if self.is_odd(diff):
                        slice_copy = slice_copy[diff // 2: -diff//2, :]
                    else:
                        slice_copy = slice_copy[diff // 2: -diff // 2, :]

                if slice.shape[1] < to_slice_shape[1]:
                    diff = to_slice_shape[1] - slice.shape[1]
                    if self.is_odd(diff):
                        slice_copy = cv2.copyMakeBorder(slice_copy, 0, 0, diff // 2, diff // 2 + 1,
                                                        cv2.BORDER_CONSTANT,
                                                        value=0.0)
                    else:
                        slice_copy = cv2.copyMakeBorder(slice_copy, 0, 0, diff // 2, diff // 2,
                                                        cv2.BORDER_CONSTANT,
                                                        value=0.0)
                elif slice.shape[1] > to_slice_shape[1]:
                    diff = slice.shape[1] - to_slice_shape[1]
                    if self.is_odd(diff):
                        slice_copy = slice_copy[:, diff // 2: -diff // 2]
                    else:
                        slice_copy = slice_copy[:, diff // 2: -diff // 2]

                resized_image.append(slice_copy)

            resized_list.append(np.asanyarray(resized_image))

        return resized_list


    def is_odd(self, number):

        return number % 2 != 0

    def threed_resize(self, image, slice_shape):

        all_slices = []
        for index in range(image.shape[2]):
            slice = image[:, :, index]
            resized = cv2.resize(slice, (slice_shape[1], slice_shape[0]), cv2.INTER_CUBIC)
            all_slices.append(resized)

        return np.asanyarray(all_slices)

    def normalize_minmax(self, images_list):

        normalized_list = []

        for image in images_list:
            section_max = np.max(image)
            section_min = np.min(image)
            normalized = (image - section_min) / (section_max - section_min)
            normalized_list.append(normalized)

        return normalized_list


    def get_first_last_slice(self, mask):

        first_slice_brain = np.argmax(mask) // np.prod(mask.shape[1:])
        last_slice_brain = (np.prod(mask.shape) - np.argmax(np.flip(mask, axis=0))) // np.prod(mask.shape[1:])

        return first_slice_brain, last_slice_brain


    def normalize_neg_pos_one(self, images_list, slice_number):

        normalized_list = []

        np_list = np.asanyarray(images_list)
        for image_idx in range(np_list.shape[0] // slice_number):
            this_section = np_list[image_idx * slice_number:(image_idx + 1) * slice_number, :, :]
            normalized = 2. * (this_section - np.min(this_section)) / np.ptp(this_section) - 1
            normalized_list.append(normalized)

        normalized_list = np.concatenate(normalized_list)
        return normalized_list



    def normalize_quantile(self, flair_list, slice_number):

        normalized_images = []
        flair_list = np.asanyarray(flair_list)

        for image_idx in range(flair_list.shape[0] // slice_number):
            this_flair = flair_list[image_idx * slice_number:(image_idx + 1) * slice_number, :, :]

            flair_non_black = this_flair[this_flair > 0]
            lower_threshold, upper_threshold, upper_indexes, lower_indexes = self.get_thresholds_and_indexes(flair_non_black,
                                                                                                             99.7,
                                                                                                             this_flair)
            final_normalized = (this_flair - lower_threshold) / (upper_threshold - lower_threshold)
            final_normalized[upper_indexes] = 1.0
            final_normalized[lower_indexes] = 0.0
            normalized_images.append(final_normalized)

        normalized_images = np.concatenate(normalized_images, axis=0)
        return normalized_images

    def standarize(self, image_list):

        standarized_imgs = []

        for image in image_list:
            stand_image = (image - np.mean(image)) / np.std(image)
            standarized_imgs.append(stand_image)

        return standarized_imgs

    def remove_third_label(self, labels_list):

        new_labels_list = []

        for image in labels_list:
            image[image > 1.] = 0.0
            new_labels_list.append(image)

        return new_labels_list
