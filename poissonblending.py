#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import PIL.Image
import pyamg

import cv2
import unsharp_mask


# pre-process the mask array so that uint64 types from opencv.imread can be adapted
def prepare_mask(mask):
    if type(mask[0][0]) is np.ndarray:
        result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if sum(mask[i][j]) > 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        mask = result
    return mask


def blend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
    region_source = (
        max(-offset[0], 0),
        max(-offset[1], 0),
        min(img_target.shape[0] - offset[0], img_source.shape[0]),
        min(img_target.shape[1] - offset[1], img_source.shape[1]))
    region_target = (
        max(offset[0], 0),
        max(offset[1], 0),
        min(img_target.shape[0], img_source.shape[0] + offset[0]),
        min(img_target.shape[1], img_source.shape[1] + offset[1]))
    region_size = (region_source[2] - region_source[0], region_source[3] - region_source[1])

    print("region source:", region_source)
    print("region target:", region_target)
    print("region size:", region_size)

    # clip and normalize mask image
    img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask = prepare_mask(img_mask)
    img_mask[img_mask == 0] = False
    img_mask[img_mask != False] = True

    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y, x]:
                index = x + y * region_size[1]
                A[index, index] = 4
                if index + 1 < np.prod(region_size):
                    A[index, index + 1] = -1
                if index - 1 >= 0:
                    A[index, index - 1] = -1
                if index + region_size[1] < np.prod(region_size):
                    A[index, index + region_size[1]] = -1
                if index - region_size[1] >= 0:
                    A[index, index - region_size[1]] = -1
    A = A.tocsr()

    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer]
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3], num_layer]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y, x]:
                    index = x + y * region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A, b, verb=False, tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x > 255] = 255
        x[x < 0] = 0
        x = np.array(x, img_target.dtype)
        img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer] = x

    return img_target


def test():
    img_mask = np.asarray(PIL.Image.open('./testimages/test1_mask.png'))
    img_mask.flags.writeable = True
    img_source = np.asarray(PIL.Image.open('./testimages/test1_src.png'))
    img_source.flags.writeable = True
    img_target = np.asarray(PIL.Image.open('./testimages/test2_opencv.png'))
    img_target.flags.writeable = True
    # img_ret = blend(img_target, img_source, img_mask, offset=(40,-30))
    img_ret = blend(img_target, img_source, img_mask, offset=(0, 0))
    img_ret = PIL.Image.fromarray(np.uint8(img_ret))

    img_ret.save('./testimages/test22_ret.png')


    src = cv2.imread('./testimages/test1_src.png')
    tar = cv2.imread('./testimages/test2_opencv.png')
    aaa = cv2.imread('./testimages/test22_ret.png')
    cv2.imshow("src", src)
    cv2.imshow("tar", tar)
    cv2.imshow("adsad", aaa)
    cv2.waitKey(0)


def inpainting():
    img = cv2.imread('./testimages/test1_src.png')
    mask = cv2.imread('./testimages/test1_mask.png', 0)

    dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

    cv2.imshow('src', img)
    cv2.imshow('mask', mask)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite("test2_opencv.png", dst)


def alpha_blending():
    # Read the images
    foreground = cv2.imread('./testimages/test1_src.png')
    background = cv2.imread('./testimages/test1_target.png')
    alpha = cv2.imread('./testimages/test1_mask.png')

    cv2.imshow("src_0", foreground)
    cv2.imshow("target_0", background)
    cv2.imshow("alpha_0", alpha)

    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)

    # Display image
    cv2.imshow("src", foreground / 255)
    cv2.imshow("outImg", outImage / 255)
    cv2.waitKey(0)


def alpha_blending_test_copy():
    # Read the images
    foreground = cv2.imread('D:/RAIN_DATA_TRAINING/GOPR0039_taken_640_broken_center_short/mask.jpg')
    background = cv2.imread('D:/RAIN_DATA_TRAINING/GOPR0039_taken_640_broken_center_short/raindrop0368.jpg')
    alpha = cv2.imread('D:/RAIN_DATA_TRAINING/GOPR0039_taken_640_broken_center_short/mask.jpg')




    cv2.imshow("src_0", foreground)
    cv2.imshow("target_0", background)
    cv2.imshow("alpha_0", alpha)

    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)

    # Display image
    cv2.imshow("src", foreground / 255)
    cv2.imshow("outImg", outImage / 255)
    cv2.waitKey(0)


def seamless_cloning(mode = 'mixed_clone'):
    # Read images : src image will be cloned into dst
    im = cv2.imread('./rain/i_raindrop0351.jpg')
    obj = cv2.imread('./rain/o_raindrop0351.jpg')


    # Create an all white mask
    mask = cv2.imread('./rain/m3_raindrop0351.jpg')

    # Create an all white mask
    # mask = 255 * np.ones(obj.shape, obj.dtype)

    # The location of the center of the src in the dst
    width, height, channels = im.shape
    center = (int(height / 2), int(width / 2))
    print("center", center)

    cv2.circle(im, center, 3, (0,0,255), -1)
    cv2.circle(obj, center, 3, (0,0,255), -1)
    cv2.circle(mask, center, 3, (0,0,255), -1)


    # dest_cloned = im
    #
    # # Seamlessly clone src into dst and put the results in output
    # if mode == 'normal_clone':
    #     dest_cloned = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
    # elif mode == 'mixed_clone':
    #     dest_cloned = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
    # else:
    #     dest_cloned= cv2.seamlessClone(obj, im, mask, center, cv2.MONOCHROME_TRANSFER)

    normal_clone = unsharp_mask.unsharp_mask(cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE))
    mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
    monochrome_transfer = cv2.seamlessClone(obj, im, mask, center, cv2.MONOCHROME_TRANSFER)

    cv2.circle(normal_clone, center, 3, (0,255,255), -1)
    cv2.circle(mixed_clone, center, 3, (0,255,255), -1)
    cv2.circle(monochrome_transfer, center, 3, (0,255,255), -1)
    # Display image
    cv2.imshow("im", im)
    cv2.imshow("obj", obj)
    cv2.imshow("mask", mask)
    cv2.imshow("normal_clone", normal_clone)
    cv2.imshow("mixed_clone", mixed_clone)
    cv2.imshow("monochrome_transfer", monochrome_transfer)

    # cv2.imshow("dest_cloned", dest_cloned)
    cv2.waitKey(0)

    # Write results
    # cv2.imwrite("result/opencv-normal-clone-example.jpg", normal_clone)
    # cv2.imwrite("result/opencv-mixed-clone-example.jpg", mixed_clone)
    # cv2.imwrite("result/opencv-monochrome-transfer-example.jpg", monochrome_transfer)


def seamless_cloning_paper(mode = 'mixed_clone'):
    # Read images : src image will be cloned into dst
    # im = cv2.imread('./code_seamless/input.jpg')
    # obj = cv2.imread('./code_seamless/output.jpg')

    im = cv2.imread('./code_seamless/newbg.jpg')
    obj = cv2.imread('./rain/i_raindrop0351.jpg')

    # Create an all white mask
    # mask = cv2.imread('./code_seamless/mask.jpg')

    mask = cv2.imread('./rain/m_raindrop0351.jpg')




    gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # thresh = 127
    # im_bw = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)[1]

    (thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("gray_image", gray_image)
    cv2.imshow("im_bw", im_bw)


    # Create an all white mask
    # mask = 255 * np.ones(obj.shape, obj.dtype)

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
    print(",minx: ", minx, ",maxx: ", maxx, ",miny: ", miny, ",maxy: ", maxy)

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


    # dest_cloned = im
    #
    # # Seamlessly clone src into dst and put the results in output
    # if mode == 'normal_clone':
    #     dest_cloned = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
    # elif mode == 'mixed_clone':
    #     dest_cloned = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
    # else:
    #     dest_cloned= cv2.seamlessClone(obj, im, mask, center, cv2.MONOCHROME_TRANSFER)

    normal_clone = cv2.seamlessClone(obj, im, im_bw, center, cv2.NORMAL_CLONE)
    normal_clone = unsharp_mask.unsharp_mask(normal_clone)

    mixed_clone = cv2.seamlessClone(obj, im, im_bw, center, cv2.MIXED_CLONE)
    mixed_clone = unsharp_mask.unsharp_mask(mixed_clone)

    monochrome_transfer = cv2.seamlessClone(obj, im, im_bw, center, cv2.MONOCHROME_TRANSFER)

    cv2.circle(normal_clone, center, 1, (0,255,255), -1)
    cv2.circle(mixed_clone, center, 1, (0,255,255), -1)
    cv2.circle(monochrome_transfer, center, 1, (0,255,255), -1)

    # Display image
    cv2.imshow("im", im)
    cv2.imshow("obj", obj)
    cv2.imshow("mask", mask)
    cv2.imshow("normal_clone", normal_clone)
    cv2.imshow("mixed_clone", mixed_clone)
    cv2.imshow("monochrome_transfer", monochrome_transfer)

    # cv2.imshow("dest_cloned", dest_cloned)
    cv2.waitKey(0)

    # Write results
    # cv2.imwrite("result/opencv-normal-clone-example.jpg", normal_clone)
    # cv2.imwrite("result/opencv-mixed-clone-example.jpg", mixed_clone)
    # cv2.imwrite("result/opencv-monochrome-transfer-example.jpg", monochrome_transfer)


def seamless_cloning_paper_test_copy(mode = 'mixed_clone'):
    # Read images : src image will be cloned into dst
    # im = cv2.imread('./code_seamless/input.jpg')
    # obj = cv2.imread('./code_seamless/output.jpg')

    im = cv2.imread('D:/RAIN_DATA_TRAINING/GOPR0039_taken_640_broken_center_short/raindrop0439.jpg')
    obj = cv2.imread('D:/RAIN_DATA_TRAINING/GOPR0039_taken_640_broken_center_short/raindrop0436.jpg')

    # Create an all white mask
    # mask = cv2.imread('./code_seamless/mask.jpg')

    mask = cv2.imread('D:/RAIN_DATA_TRAINING/GOPR0039_taken_640_broken_center_short/mask01.jpg')


    crop_img = obj[370:470, 270:370]
    cv2.imshow("cropped", crop_img)

    x_offset = y_offset = 270
    im[y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = crop_img

    cv2.imshow("im", im)
    #cv2.waitKey(0)

    gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # thresh = 127
    # im_bw = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)[1]

    (thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("gray_image", gray_image)
    cv2.imshow("im_bw", im_bw)
    cv2.imwrite("D:/RAIN_DATA_TRAINING/GOPR0039_taken_640_broken_center_short/copied_image_439_436.jpg", im)


    # Create an all white mask
    # mask = 255 * np.ones(obj.shape, obj.dtype)

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
    print(",minx: ", minx, ",maxx: ", maxx, ",miny: ", miny, ",maxy: ", maxy)

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


    # dest_cloned = im
    #
    # # Seamlessly clone src into dst and put the results in output
    # if mode == 'normal_clone':
    #     dest_cloned = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
    # elif mode == 'mixed_clone':
    #     dest_cloned = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
    # else:
    #     dest_cloned= cv2.seamlessClone(obj, im, mask, center, cv2.MONOCHROME_TRANSFER)
    taolao = (320, 420)
    normal_clone = cv2.seamlessClone(obj, im, im_bw, center, cv2.NORMAL_CLONE)
    normal_clone = unsharp_mask.unsharp_mask(normal_clone)

    mixed_clone = cv2.seamlessClone(obj, im, im_bw, center, cv2.MIXED_CLONE)
    mixed_clone = unsharp_mask.unsharp_mask(mixed_clone)

    monochrome_transfer = cv2.seamlessClone(obj, im, im_bw, center, cv2.MONOCHROME_TRANSFER)

    cv2.circle(normal_clone, center, 1, (0,255,255), -1)
    cv2.circle(mixed_clone, center, 1, (0,255,255), -1)
    cv2.circle(monochrome_transfer, center, 1, (0,255,255), -1)

    # Display image
    cv2.imshow("im", im)
    cv2.imshow("obj", obj)
    cv2.imshow("mask", mask)
    cv2.imshow("normal_clone", normal_clone)
    cv2.imshow("mixed_clone", mixed_clone)
    cv2.imshow("monochrome_transfer", monochrome_transfer)

    # cv2.imshow("dest_cloned", dest_cloned)
    cv2.waitKey(0)

    # Write results
    # cv2.imwrite("result/opencv-normal-clone-example.jpg", normal_clone)
    # cv2.imwrite("result/opencv-mixed-clone-example.jpg", mixed_clone)
    # cv2.imwrite("result/opencv-monochrome-transfer-example.jpg", monochrome_transfer)


if __name__ == '__main__':
    # inpainting()
    # test()
    # alpha_blending()
    # seamless_cloning_paper()
    seamless_cloning_paper_test_copy()
    # alpha_blending_test_copy()



