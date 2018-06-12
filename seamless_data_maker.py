#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import PIL.Image
import pyamg

import os

import Augmentor

import cv2
import unsharp_mask


def seamless_cloning_paper(im_path = 'seamless_data_maker/im/raindrop0653.jpg', obj_path = 'seamless_data_maker/obj/obj.jpg',
                           mask_path = 'seamless_data_maker/mask/mask.jpg', mode = 'mixed_clone'):
    # Read images : src image will be cloned into dst

    im = cv2.imread(im_path)
    obj = cv2.imread(obj_path)

    mask = cv2.imread(mask_path)
    origin_mask = mask.copy()

    gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    (thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # The location of the center of the src in the dst
    width, height, channels = im.shape

    minx = 1e5
    maxx = 1
    miny = 1e5
    maxy = 1

    for y in range(1, height):
        for x in range(1, width):
            if ((im_bw[x][y] != 0) and (im_bw[x][y] != 255)):
                print(x, y , " : ",im_bw[x][y])
            # if ((mask[x][y][1] > 0) or (mask[x][y][2] > 0) or (mask[x][y][0] > 0) ):
            if ((im_bw[x][y] > 0) ):
                # print(x, y , " : ",im_bw[x][y])

                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)

    center = (int(miny+(maxy-miny)/2) , int(minx+(maxx-minx)/2))

    # center = (int(height / 2), int(width / 2))
    print("center", center)
    print("minx: ", minx, ",maxx: ", maxx, ",miny: ", miny, ",maxy: ", maxy)

    cv2.circle(im, center, 1, (0,0,255), -1)
    cv2.circle(obj, center, 1, (0,0,255), -1)
    cv2.circle(mask, center, 1, (0,0,255), -1)

    # draw points
    cv2.circle(mask,(minx, miny), 1, (0,255,255), -1)
    cv2.circle(mask,(maxx, miny), 1, (0,255,255), -1)
    cv2.circle(mask,(minx, maxy), 1, (0,255,255), -1)
    cv2.circle(mask,(maxx, maxy), 1, (0,255,255), -1)

    # draw points
    cv2.circle(mask,(miny, minx), 1, (0,255,0), -1)
    cv2.circle(mask,(maxy, minx), 1, (0,255,0), -1)
    cv2.circle(mask,(miny, maxx), 1, (0,255,0), -1)
    cv2.circle(mask,(maxy, maxx), 1, (0,255,0), -1)
    cv2.rectangle(mask,(miny, minx),(maxy, maxx),(0,255,0),1)

    normal_clone = cv2.seamlessClone(obj, im, im_bw, center, cv2.NORMAL_CLONE)
    # normal_clone = unsharp_mask.unsharp_mask(normal_clone)

    mixed_clone = cv2.seamlessClone(obj, im, im_bw, center, cv2.MIXED_CLONE)
    # mixed_clone = unsharp_mask.unsharp_mask(mixed_clone)

    monochrome_transfer = cv2.seamlessClone(obj, im, im_bw, center, cv2.MONOCHROME_TRANSFER)

    cv2.circle(normal_clone, center, 1, (0,255,255), -1)
    cv2.circle(mixed_clone, center, 1, (0,255,255), -1)
    cv2.circle(monochrome_transfer, center, 1, (0,255,255), -1)

    # Display image
    # cv2.imshow("im", im)
    # cv2.imshow("obj", obj)
    # cv2.imshow("mask", mask)
    # cv2.imshow("normal_clone", normal_clone)
    # cv2.imshow("mixed_clone", mixed_clone)
    # cv2.imshow("monochrome_transfer", monochrome_transfer)
    #
    # cv2.waitKey(30)

    # Write results
    file_name_string = im_path.split("/")
    new_file_name = os.path.basename(obj_path) + "_"+ file_name_string[-2] + "_" + file_name_string[-1]

    print(new_file_name)

    cv2.imwrite("seamless_data_maker/im_result/result/" + new_file_name, normal_clone)
    cv2.imwrite("seamless_data_maker/im_result/mask/" + new_file_name, origin_mask)


    # cv2.imwrite("result/opencv-normal-clone-example.jpg", normal_clone)
    # cv2.imwrite("result/opencv-mixed-clone-example.jpg", mixed_clone)
    # cv2.imwrite("result/opencv-monochrome-transfer-example.jpg", monochrome_transfer)


if __name__ == '__main__':

    # seamless_cloning_paper()
    path = "seamless_data_maker/im"
    training_dirs = os.listdir(path)
    print(training_dirs)

    training_file_names  = []

    # append all files into 2 lists
    for training_dir in training_dirs:
        # append each file into the list file names
        training_folder = os.listdir(path + "/" + training_dir)
        for training_item in training_folder:
            # modify to full path -> directory
            training_item = path + "/" + training_dir + "/" + training_item
            training_file_names.append(training_item)

    path_source_obj = "seamless_data_maker/source/obj"
    source_files = os.listdir(path_source_obj)

    for obj in source_files:
        # print all file paths
        for i in training_file_names:
            print(i)
            # use seamless cloning
            # seamless_cloning_paper(im_path=i,
            #                        obj_path= "seamless_data_maker/source/obj/" + obj,
            #                        mask_path="seamless_data_maker/source/mask/" + obj)

    # Generate
    p = Augmentor.Pipeline(source_directory="/media/sf_shared/poissonblending/seamless_data_maker/im_result/result",
                           output_directory="/media/sf_shared/poissonblending/seamless_data_maker/output")
    # Point to a directory containing ground truth data.
    # Images with the same file names will be added as ground truth data
    # and augmented in parallel to the original data.
    p.ground_truth("/media/sf_shared/poissonblending/seamless_data_maker/im_result/mask")
    # Add operations to the pipeline as normal:
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.flip_top_bottom(probability=0.5)
    p.sample(2)

    # Move and rename
    output_path = "seamless_data_maker/output"
    output_ready_path = "seamless_data_maker/output_ready"

    output_folder = os.listdir(output_path)
    output_folder.sort()

    # move
    for file in output_folder:
        base_file_name = os.path.basename(file)

        if base_file_name.find("_groundtruth_") > -1:
            # add "mask"
            new_base_file_name = base_file_name.replace("_groundtruth_(1)_result_", "")
            new_base_file_name = new_base_file_name[:-4] + "_mask" + new_base_file_name[-4:]
            print(new_base_file_name)

            os.rename(output_path + "/" + base_file_name, output_ready_path + "/train_masks/" + new_base_file_name)
        else:
            new_base_file_name = base_file_name.replace("result_original_", "")
            print(new_base_file_name)

            os.rename(output_path + "/" + base_file_name, output_ready_path + "/train/" + new_base_file_name)

