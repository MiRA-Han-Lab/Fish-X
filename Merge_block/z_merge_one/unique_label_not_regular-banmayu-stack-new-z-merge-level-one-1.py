import numpy as np


import tifffile
from libtiff import TIFF
import imageio
import math

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000
import os, sys
import struct
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from numba import njit
from scipy.ndimage import affine_transform



import scipy.io as sio
from scipy.ndimage import zoom

import h5py
import glob
import fastremap

from collections import defaultdict

import fast64counter
import imageio
import time
import matplotlib.pyplot as plth
import networkx as nx

import torch
import time

#import cc3d

def overlap_fusing(img1, img2, direction, halo_size, whether_label):
    if whether_label == False:

        if direction == 3:
            for num in range(0, img2.shape[0]):

                for j in range(halo_size):
                    alpha = (halo_size - j) / halo_size
                    img2[num, :, j] = alpha * img1[num, :, img1.shape[2] - halo_size + j] + (1 - alpha) * img2[num, :,
                                                                                                          j]

        if direction == 2:
            for num in range(0, img2.shape[0]):

                for j in range(halo_size):
                    alpha = (halo_size - j) / halo_size
                    img2[num, j, :] = alpha * img1[num, img1.shape[1] - halo_size + j, :] + (1 - alpha) * img2[num, j,
                                                                                                          :]

        if direction == 1:
            for j in range(halo_size):
                alpha = (halo_size - j) / halo_size
                img2[j, :, :] = alpha * img1[img1.shape[0] - halo_size + j, :, :] + (1 - alpha) * img2[j, :, :]

    else:
        pass

    return img1, img2


def make_label_s(block1, block2, label, height, hang, line, direction):
    
    
    
    
    
    

    

    
    
    
    

    
    
    
    

    
    
    
    

    uq, s, p = np.unique(block2, return_index=True, return_inverse=True)
    label1_max = np.max(block1)
    index = p.reshape(block2.shape)
    label_new = np.arange(start=label1_max + 1, stop=label1_max + 1 + uq.shape[0], step=1, dtype='int32')
    label_new[uq == 0] = 0
    block2_new = label_new[index]

    s0 = s // (block2.shape[1] * block2.shape[2])
    s0_yu = s % (block2.shape[1] * block2.shape[2])
    s1 = s0_yu // block2.shape[1]  
    s2 = s0_yu % block2.shape[1]  
    

    
    

    
    
    
    

    if direction == 3:
        mark = np.ones(label.shape[3], dtype=bool)
        for i, j in zip(block2[s0, s1, s2], block2_new[s0, s1, s2]):
            whether = (label[height][hang][line] == i)
            whether = np.logical_and(mark, whether)
            label[height][hang][line][whether] = j
            mark[whether == True] = False

    elif direction == 2:
        for num_line in range(0, line):
            mark = np.ones(label.shape[3], dtype=bool)
            for i, j in zip(block2[s0, s1, s2], block2_new[s0, s1, s2]):
                whether = (label[height][hang][line] == i)
                whether = np.logical_and(mark, whether)
                label[height][hang][line][whether] = j
                mark[whether == True] = False

    elif direction == 1:
        for num_hang in range(0, hang):
            for num_line in range(0, line):
                mark = np.ones(label.shape[3], dtype=bool)
                for i, j in zip(block2[s0, s1, s2], block2_new[s0, s1, s2]):
                    whether = (label[height][hang][line] == i)
                    whether = np.logical_and(mark, whether)
                    label[height][hang][line][whether] = j
                    mark[whether == True] = False

    return block1, block2_new, label


def make_label(block1, block2, max_block1_part_all, max_block2_part_all, label, height, hang, line, direction):
    
    
    label1_max = max_block1_part_all

    
    block2_new = np.array(block2 + label1_max)
    block2_new = np.where(block2_new == label1_max, 0, block2_new)

    if direction == 3: 
        label[height][hang][line] = label[height][hang][line] + label1_max
        label[height][hang][line] = np.where(label[height][hang][line] == label1_max, 0, label[height][hang][line])


    elif direction == 2:  
        for num_line in range(0, line):
            label[height][hang][num_line] = label[height][hang][num_line] + label1_max
            label[height][hang][num_line] = np.where(label[height][hang][num_line] == label1_max, 0,
                                                     label[height][hang][num_line])

    elif direction == 1: 
        for num_hang in range(0, hang):
            for num_line in range(0, line):
                label[height][num_hang][num_line] = label[height][num_hang][num_line] + label1_max
                label[height][num_hang][num_line] = np.where(label[height][num_hang][num_line] == label1_max, 0,
                                                             label[height][num_hang][num_line])

    return block1, block2_new, label


def make_label_new(block1, block2):
    label1_max = np.max(block1)

    block2_new = [x + label1_max for x in block2]
    
    block2_new = np.where(block2_new == label1_max, 0, block2_new)

    
    
    
    
    

    return block1, block2_new


def unique_label(label1_max, block2):
    label2, index = np.unique(block2, return_inverse=True)
    index = index.reshape(block2.shape)
    label2_new = np.arange(start=label1_max + 1, stop=label1_max + 1 + label2.shape[0], step=1, dtype='uint64')
    label2_new[label2 == 0] = 0
    block2_new = label2_new[index]

    return block2_new


def mul_tif_read(tif_loc):
    tif = TIFF.open(tif_loc, mode='r')
    imgs = np.array(list(tif.iter_images()))

    
    

    assert imgs.dtype == np.uint8, 'Need to be uint8 of numpy array.'
    assert np.max(np.logical_and(imgs < 255, imgs > 0)) == False, 'Need to be segmentation with only 0 and 255.'

    return imgs


def get_one_block_size(load_path, whether_loadpath=True):
    print('load_path: ' + load_path)

    if whether_loadpath == True:
        data_files = os.listdir(load_path)
        data_path = os.path.join(load_path, data_files[0])

        h5_files = os.listdir(data_path)
        data_path = os.path.join(data_path, h5_files[0])

    else:

        h5_files = os.listdir(load_path)
        data_path = os.path.join(load_path, h5_files[0])
        

    print('data_path: ' + data_path)

    if len(data_path.split('.')) == 1:

        images = os.listdir(data_path)

        if len(images) > 10:
            size_z = len(images)
            image_name = data_path + '/' + images[0]  
            image_temp = read_32_tif(image_name)
            size_y = image_temp.shape[0]
            size_x = image_temp.shape[1]
        else:
            file_name = []
            if 'seg_inv.h5' in images:
                file_name = 'seg_inv.h5'
            else:
                file_name = 'seg_inv.tif'

            if file_name.startswith('seg'):
                if file_name.endswith('.h5'):
                    with h5py.File(os.path.join(data_path, file_name), 'r') as f:
                        
                        size_z_y_x = f['shape'][:]
                elif file_name.endswith('.tif'):
                    labels = imageio.volread(os.path.join(data_path, 'seg.tif'))
                    if len(labels.shape) == 4:
                        labels = labels[:, 0, :, :]
                    elif len(labels.shape) == 3:
                        pass
                    else:
                        raise NameError
                else:
                    raise NameError
                
            else:
                raise NameError
            
            
            
            
            

            size_z = int(size_z_y_x[0])
            size_y = int(size_z_y_x[1])
            size_x = int(size_z_y_x[2])
    else:

        if data_path.split('.')[1] == 'h5':
            file_name = data_path
            
            

            
            
            

            with h5py.File(file_name, 'r') as f:
                
                size_z_y_x = f['shape'][:]
            size_z = int(size_z_y_x[0])
            size_y = int(size_z_y_x[1])
            size_x = int(size_z_y_x[2])
            print(file_name, ' size_z: ', size_z, ' size_y: ', size_y, ' size_x: ', size_x)

        else:
            raise NameError

    return size_z, size_y, size_x


def get_one_block_size_default(load_path, whether_loadpath=True):
    size_z = 55
    size_y = 2750
    size_x = 2750

    return size_z, size_y, size_x


def read_32_tif(name):
    raw = imageio.imread(name)
    return raw


def read_32_tif_seqqence(image_dir):
    test_data_path = os.path.join(image_dir)
    images = os.listdir(test_data_path)
    if len(images) > 10:
        images.sort(key=lambda x: int(x.split('.')[0]))  
        total = int(len(images))  

        image_name = image_dir + '/' + images[0]  
        image_temp = read_32_tif(image_name)
        size_y = image_temp.shape[0]
        size_x = image_temp.shape[1]
        img = np.zeros((total, size_y, size_x), dtype=image_temp.dtype)
        del image_temp

        for i, filename in enumerate(images):
            image_name = image_dir + '/' + str(filename)  
            image_temp = read_32_tif(image_name)
            img[i] = image_temp

        print('reading over')
    else:
        image_name = image_dir + '/' + images[0]  

        with h5py.File(image_name, 'r') as f:
            img = f['data'][:]
            size_z_y_x = f['shape'][:]

        img = np.int32(img)  


        size_z = int(size_z_y_x[0])
        size_y = int(size_z_y_x[1])
        size_x = int(size_z_y_x[2])
        print(image_name, ' size_z: ', size_z, ' size_y: ', size_y, ' size_x: ', size_x)

    return img


def read_chen_z_y_x(filename):
    suffix = 'jpg'
    filename_text = filename.split(".")[0]

    name = (filename_text.rsplit("/", 1)[1])
    address = (filename_text.rsplit("/", 1)[0])
    

    order_start = int(name.split("-")[0])
    order_end = int(name.split("_")[0].split("-")[1])
    
    order_layer = int(order_end) - int(order_start) + 1
    
    order_y = int(filename_text.split("_")[-2])
    order_x = int(filename_text.split("_")[-1])

    print('example: ',
          address + '/' + 'layer' + str(order_start) + '_' + str(order_y) + '_' + str(order_x) + '.' + suffix)

    example = cv2.imread(
        address + '/' + 'layer' + str(order_start) + '_' + str(order_y) + '_' + str(order_x) + '.' + suffix,
        cv2.IMREAD_GRAYSCALE)

    data = np.zeros((int(order_layer), example.shape[0], example.shape[1]), dtype=np.uint8)

    del example

    for i in range(int(order_layer)):
        image_name = address + '/' + 'layer' + str(i + int(order_start)) + '_' + str(order_y) + '_' + str(
            order_x) + '.' + suffix  
        image_temp = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

        data[i, :, :] = image_temp
    print('reading over')

    return data


def read_32_tif_seqqence_part(image_dir, whether_line, whether_row, whether_height, halo_size, front, label, height,
                              hang, line):
    print('image_dir: ' + image_dir)

    h5_files = os.listdir(image_dir)
    image_dir = os.path.join(image_dir, h5_files[0])

    test_data_path = os.path.join(image_dir)

    if len(image_dir.split('.')) == 1:

        images = os.listdir(test_data_path)

        if len(images) > 10:

            images.sort(key=lambda x: int(x.split('.')[0]))  
            sorted_file = images
            total = int(len(images))  

            img = []
            if whether_line == True:
                if front == True:
                    for i, filename in enumerate(sorted_file):
                        image_name = image_dir + '/' + str(filename)  

                        with Image.open(image_name) as pil_image:
                            
                            kuan = pil_image.size[0]  
                            chang = pil_image.size[1]  

                            image_temp = np.array(
                                pil_image.crop([kuan - halo_size, 0, kuan, chang]))  
                            image_temp = np.expand_dims(image_temp, 0)
                            if i == 0:
                                img = image_temp
                            else:
                                img = np.concatenate([img, image_temp], axis=0)
                else:
                    for i, filename in enumerate(sorted_file):
                        image_name = image_dir + '/' + str(filename)  

                        with Image.open(image_name) as pil_image:
                            
                            kuan = pil_image.size[0]  
                            chang = pil_image.size[1]  

                            image_temp = np.array(pil_image.crop([0, 0, halo_size, chang]))  
                            image_temp = np.expand_dims(image_temp, 0)
                            if i == 0:
                                img = image_temp
                            else:
                                img = np.concatenate([img, image_temp], axis=0)

            if whether_row == True:
                if front == True:
                    for i, filename in enumerate(sorted_file):
                        image_name = image_dir + '/' + str(filename)  

                        with Image.open(image_name) as pil_image:
                            
                            kuan = pil_image.size[0]  
                            chang = pil_image.size[1]  

                            image_temp = np.array(
                                pil_image.crop([0, chang - halo_size, kuan, chang]))  
                            image_temp = np.expand_dims(image_temp, 0)
                            if i == 0:
                                img = image_temp
                            else:
                                img = np.concatenate([img, image_temp], axis=0)
                else:
                    for i, filename in enumerate(sorted_file):
                        image_name = image_dir + '/' + str(filename)  

                        with Image.open(image_name) as pil_image:
                            
                            kuan = pil_image.size[0]  
                            chang = pil_image.size[1]  

                            image_temp = np.array(pil_image.crop([0, 0, kuan, halo_size]))  
                            image_temp = np.expand_dims(image_temp, 0)
                            if i == 0:
                                img = image_temp
                            else:
                                img = np.concatenate([img, image_temp], axis=0)

            if whether_height == True:
                if front == True:
                    num = len(sorted_file)
                    for i, filename in enumerate(sorted_file):
                        if i < num - halo_size:
                            continue
                        else:
                            image_name = image_dir + '/' + str(filename)  

                            with Image.open(image_name) as pil_image:
                                
                                kuan = pil_image.size[0]  
                                chang = pil_image.size[1]  

                                image_temp = np.array(pil_image.crop([0, 0, kuan, chang]))  
                                image_temp = np.expand_dims(image_temp, 0)
                                if i == num - halo_size:
                                    img = image_temp
                                else:
                                    img = np.concatenate([img, image_temp], axis=0)
                else:
                    for i, filename in enumerate(sorted_file):
                        image_name = image_dir + '/' + str(filename)  

                        with Image.open(image_name) as pil_image:
                            
                            kuan = pil_image.size[0]  
                            chang = pil_image.size[1]  

                            image_temp = np.array(pil_image.crop([0, 0, kuan, chang]))  
                            image_temp = np.expand_dims(image_temp, 0)
                            if i == 0:
                                img = image_temp
                            else:
                                img = np.concatenate([img, image_temp], axis=0)

                        if i == halo_size - 1:
                            break
        else:
            file_name = []
            if 'seg_inv.h5' in images:
                file_name = 'seg_inv.h5'
            else:
                file_name = 'seg.tif'

            if file_name.startswith('seg'):
                if file_name.endswith('.h5'):
                    with h5py.File(os.path.join(image_dir, file_name), 'r') as f:
                        labels = f['data'][:]
                        size_z_y_x = f['shape'][:]

                elif file_name.endswith('.tif'):
                    labels = imageio.volread(os.path.join(image_dir, 'seg.tif'))
                if len(labels.shape) == 4:
                    labels = labels[:, 0, :, :]
                elif len(labels.shape) == 3:
                    pass
                else:
                    raise NameError
                labels = np.transpose(labels, (2, 1, 0))
            else:
                raise NameError

            
            labels = np.int32(labels)
            

            
            labels = label_update(labels, label, height, hang, line)  

            minval, maxval = fastremap.minmax(labels)

            size_z = labels.shape[0]
            size_y = labels.shape[1]
            size_x = labels.shape[2]
            try:
                assert size_z == int(size_z_y_x[0])
                assert size_y == int(size_z_y_x[1])
                assert size_x == int(size_z_y_x[2])
            except:
                print(image_dir, ':', labels.shape)

            kuan = size_x
            chang = size_y
            num = size_z

            img = []
            if whether_line == True:
                if front == True:

                    img = labels[:, :, kuan - halo_size:]

                else:

                    img = labels[:, :, :halo_size]

            if whether_row == True:
                if front == True:

                    img = labels[:, chang - halo_size:, :]

                else:

                    img = labels[:, :halo_size:, :]

            if whether_height == True:
                if front == True:

                    img = labels[num - halo_size:, :, :]

                else:

                    img = labels[:halo_size, :, :]
    else:
        file_name = image_dir
        labels = h5py.File(file_name, 'r')['data'][:]

        
        labels = np.int32(labels)
        

        
        labels = label_update(labels, label, height, hang, line)  

        minval, maxval = fastremap.minmax(labels)

        print(image_dir, ':', labels.shape)

        size_z = labels.shape[0]
        size_y = labels.shape[1]
        size_x = labels.shape[2]

        kuan = size_x
        chang = size_y
        num = size_z

        img = []
        if whether_line == True:
            if front == True:

                img = labels[:, :, kuan - halo_size:]

            else:

                img = labels[:, :, :halo_size]

        if whether_row == True:
            if front == True:

                img = labels[:, chang - halo_size:, :]

            else:

                img = labels[:, :halo_size:, :]

        if whether_height == True:
            if front == True:

                img = labels[num - halo_size:, :, :]

            else:

                img = labels[:halo_size, :, :]

    print('reading over')
    
    
    
    

    return img, maxval


@njit
def label_update(block, label, height, hang, line):
    Update = False
    for num, value in enumerate(label[height][hang][line]):
        if num != value:
            Update = True
            break
    if Update == True:
        for i in range(0, block.shape[0]):
            for j in range(0, block.shape[1]):
                for k in range(0, block.shape[2]):
                    block[i, j, k] = label[height][hang][line][block[i, j, k]]
    return block


def img_upadte(image_dir, save_path, label, height, hang, line, whether_h5=True):
    if whether_h5 == False:

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        block = read_32_tif_seqqence(image_dir)
        block = label_update(block, label, height, hang, line)
        for i in range(0, block.shape[0]):
            out_tiff = TIFF.open(save_path + '/' + str(i).zfill(4) + '.tif', mode='w')
            out_tiff.write_image(block[i, :, :], compression=None, write_rgb=True)
            out_tiff.close()
            
        print(block.shape)

    else:

        h5_files = os.listdir(image_dir)
        image_dir = os.path.join(image_dir, h5_files[0])

        block = h5py.File(image_dir, 'r')['data'][:]

        
        block = np.int32(block)  
        

        
        block = label_update(block, label, height, hang, line)
        
        
        
        

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i in range(0, block.shape[0]):
            out_tiff = TIFF.open(save_path + '/' + str(i).zfill(4) + '.tif', mode='w')
            out_tiff.write_image(block[i, :, :], compression=None, write_rgb=True)
            out_tiff.close()
        

        print(block.shape)


def img_upadte_all(FilenameAray, load_path, sorted_file, save_path, label, num_x, num_y, num_z):
    print('begin write')
    for file in sorted_file:

        
        
        

        image_dir = load_path + '/' + file
        save = save_path + '/' + file

        if not os.path.exists(save):
            os.mkdir(save)

        for i in range(0, num_z):
            for j in range(0, num_y):
                for k in range(0, num_x):
                    if FilenameAray[i][j][k] == file:
                        img_upadte(image_dir, save, label, i, j, k, whether_h5=True)
        
        
        
        
        


def img_update_from_binary(load_path, save_path, dat_path):
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    sub_data_path = os.path.join(load_path)
    sub_tifs = os.listdir(sub_data_path)
    total = int(len(sub_tifs))  
    sort_num_first = []
    for name in sub_tifs:
        sort_num_first.append(int(name.split(".")[0]))  
    sort_num_first.sort()
    sorted_file = []
    for sort_num in sort_num_first:
        for file in sub_tifs:
            if int(sort_num) == int(file.split(".")[0]):
                sorted_file.append(file)

    bite, num_z, num_y, num_x, label_new = load_binary_dat(dat_path)

    img_upadte_all(load_path, sorted_file, save_path, label_new, num_x, num_y, num_z)


def save_binary_dat(label, save_path, bite, num_z, num_y, num_x, overloap_z_pixel, overloap_y_pixel, overloap_x_pixel,
                    size_z, size_y, size_x, height, hang, line, save_name = 'merge.dat'):
    
    

    with open(save_path + '/' + save_name, "wb") as outfile:  
        num_bite_0 = struct.pack('i', bite)  
        outfile.write(num_bite_0)

        num_z_1 = struct.pack('i', num_z)  
        outfile.write(num_z_1)

        num_y_2 = struct.pack('i', num_y)  
        outfile.write(num_y_2)

        num_x_3 = struct.pack('i', num_x)  
        outfile.write(num_x_3)

        overloap_z_pixel_4 = struct.pack('i', overloap_z_pixel)  
        outfile.write(overloap_z_pixel_4)

        overloap_y_pixel_5 = struct.pack('i', overloap_y_pixel)  
        outfile.write(overloap_y_pixel_5)

        overloap_x_pixel_6 = struct.pack('i', overloap_x_pixel)  
        outfile.write(overloap_x_pixel_6)

        
        size_z_7 = struct.pack('i', size_z)  
        outfile.write(size_z_7)

        size_y_8 = struct.pack('i', size_y)  
        outfile.write(size_y_8)

        size_x_9 = struct.pack('i', size_x)  
        outfile.write(size_x_9)

        height_10 = struct.pack('i', height)  
        outfile.write(height_10)

        hang_11 = struct.pack('i', hang)  
        outfile.write(hang_11)

        line_12 = struct.pack('i', line)  
        outfile.write(line_12)

        for i in range(0, label.shape[0]):  
            for j in range(0, label.shape[1]):
                for k in range(0, label.shape[2]):
                    for m in range(0, label.shape[3]):
                        tmp = struct.pack('i', label[i, j, k, m])
                        outfile.write(tmp)


def load_binary_dat(merge):
    

    with open(merge, 'rb') as f:
        bite = struct.unpack('i', f.read(4))  
        num_z = struct.unpack('i', f.read(4))  
        num_y = struct.unpack('i', f.read(4))  
        num_x = struct.unpack('i', f.read(4))  
        overloap_z_pixel = struct.unpack('i', f.read(4))  
        overloap_y_pixel = struct.unpack('i', f.read(4))  
        overloap_x_pixel = struct.unpack('i', f.read(4))  
        size_z = struct.unpack('i', f.read(4))  
        size_y = struct.unpack('i', f.read(4))  
        size_x = struct.unpack('i', f.read(4))  
        height = struct.unpack('i', f.read(4))  
        hang = struct.unpack('i', f.read(4))  
        line = struct.unpack('i', f.read(4))  

        label = np.zeros((int(num_z[0]), int(num_y[0]), int(num_x[0]), 2 ** int(bite[0])), dtype=np.int32)

        arr_read = struct.unpack('{}i'.format((int(num_x[0]) * int(num_y[0]) * int(num_z[0])) * (2 ** int(bite[0]))),
                                 f.read())  
        arr_read = np.array(list(arr_read), dtype=np.int32)
        arr_read = arr_read.reshape(label.shape)

    return bite, num_z, num_y, num_x, arr_read


def load_binary_dat_break_down(merge):
    

    with open(merge, 'rb') as f:
        bite = struct.unpack('i', f.read(4))  
        num_z = struct.unpack('i', f.read(4))  
        num_y = struct.unpack('i', f.read(4))  
        num_x = struct.unpack('i', f.read(4))  
        overloap_z_pixel = struct.unpack('i', f.read(4))  
        overloap_y_pixel = struct.unpack('i', f.read(4))  
        overloap_x_pixel = struct.unpack('i', f.read(4))  
        size_z = struct.unpack('i', f.read(4))  
        size_y = struct.unpack('i', f.read(4))  
        size_x = struct.unpack('i', f.read(4))  
        height = struct.unpack('i', f.read(4))  
        hang = struct.unpack('i', f.read(4))  
        line = struct.unpack('i', f.read(4))  

        label = np.zeros((int(num_z[0]), int(num_y[0]), int(num_x[0]), 2 ** int(bite[0])), dtype=np.int32)

        #print(bite) ####
        #if int(bite[0]) != 19:   ####
        #    print('error here!!!!!')  ####
            
        
        
        arr_read = struct.unpack('{}i'.format((int(num_x[0]) * int(num_y[0]) * int(num_z[0])) * (2 ** int(bite[0]))),
                                 f.read())  
        #arr_read = struct.unpack('{}i'.format((int(num_x[0]) * int(num_y[0]) * int(num_z[0])) * (2 ** 19)), f.read())                           
        arr_read = np.array(list(arr_read), dtype=np.int32)
        arr_read = arr_read.reshape(label.shape)
        
                

        
        

    return int(height[0]), int(hang[0]), int(line[0]), arr_read


def read_chongdie_area_all(FilenameAray, SizeAray, load_path, num_x, num_y, whether_down, height, label,
                           overloap_x_pixel, overloap_y_pixel, halo_size):
    

    if whether_down == True:

        max_block1_part_all = 0
        max_block2_part_all = 0

        for hang in range(0, num_y):

            if FilenameAray[height - 1][hang][0] == '0':
                hang_first = np.zeros((halo_size, SizeAray[height - 1][hang][0][1], SizeAray[height - 1][hang][0][2]),
                                      dtype=np.int32)
                max_block1_part = 0

            else:
                hang_first_name = load_path + '/' + FilenameAray[height - 1][hang][0]
                hang_first, max_block1_part = read_32_tif_seqqence_part(hang_first_name, False, False, True, halo_size,
                                                                        True, label, 0, hang, 0)    
                

                if max_block1_part > max_block1_part_all:
                    max_block1_part_all = max_block1_part

            for line in range(1, num_x):

                if FilenameAray[height - 1][hang][line] == '0':

                    hang_next = np.zeros(
                        hang_first(halo_size, SizeAray[height - 1][hang][line][1], SizeAray[height - 1][hang][line][2]),
                        dtype=np.int32)
                    hang_first = np.concatenate([hang_first, hang_next[:, :, overloap_x_pixel:]], axis=2)

                    max_block1_part = 0

                else:
                    hang_next_name = load_path + '/' + FilenameAray[height - 1][hang][line]
                    hang_next, max_block1_part = read_32_tif_seqqence_part(hang_next_name, False, False, True,
                                                                           halo_size, True, label, 0, hang,   
                                                                           line)
                    

                    hang_first = np.concatenate([hang_first, hang_next[:, :, overloap_x_pixel:]], axis=2)

                    if max_block1_part > max_block1_part_all:
                        max_block1_part_all = max_block1_part

            if hang == 0:
                block1_part = hang_first
            else:
                block1_part = np.concatenate([block1_part, hang_first[:, overloap_y_pixel:, :]], axis=1)

        del hang_first

        return block1_part, max_block1_part_all

        

    else:

        max_block1_part_all = 0
        max_block2_part_all = 0

        for hang in range(0, num_y):

            if FilenameAray[height][hang][0] == '0':
                hang_first = np.zeros((halo_size, SizeAray[height][hang][0][1], SizeAray[height][hang][0][2]),
                                      dtype=np.int32)
                max_block2_part = 0

            else:
                hang_first_name = load_path + '/' + FilenameAray[height][hang][0]
                hang_first, max_block2_part = read_32_tif_seqqence_part(hang_first_name, False, False, True, halo_size,
                                                                        False, label, 0, hang, 0)     
                

                if max_block2_part > max_block2_part_all:
                    max_block2_part_all = max_block2_part

            for line in range(1, num_x):

                if FilenameAray[height][hang][line] == '0':
                    hang_next = np.zeros((halo_size, SizeAray[height][hang][line][1], SizeAray[height][hang][line][2]),
                                         dtype=np.int32)
                    hang_first = np.concatenate([hang_first, hang_next[:, :, overloap_x_pixel:]], axis=2)
                else:
                    hang_next_name = load_path + '/' + FilenameAray[height][hang][line]
                    hang_next, max_block2_part = read_32_tif_seqqence_part(hang_next_name, False, False, True,
                                                                           halo_size, False, label, 0, hang, line)     
                    

                    hang_first = np.concatenate([hang_first, hang_next[:, :, overloap_x_pixel:]], axis=2)

                    if max_block2_part > max_block2_part_all:
                        max_block2_part_all = max_block2_part

            if hang == 0:
                block2_part = hang_first
            else:
                block2_part = np.concatenate([block2_part, hang_first[:, overloap_y_pixel:, :]], axis=1)

        del hang_first

        return block2_part, max_block2_part_all


def compute_matrix(matrix_cell, i):
    if i == 1:
        matrix = matrix_cell[i - 1]
    else:
        matrix = np.linalg.multi_dot(matrix_cell[0:i])
    a_pinv = np.linalg.pinv(matrix)
    a_pinv[2, 0] = 0
    a_pinv[2, 1] = 0
    a_pinv[2, 2] = 1
    
    return a_pinv


def Affine_transformation(FilenameAray, SizeAray, block1_part, block2_part, affine_address, affine_scale, height, hang,
                          line, whether_line, whether_row, whether_height, pad_size, order):
    

    if whether_line == True:

        block1_name = FilenameAray[height][hang][line - 1].split('.')[0]
        block2_name = FilenameAray[height][hang][line].split('.')[0]

        if block1_name == '0':
            block1_out = np.pad(block1_part, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')
        else:
            pad_block1_part = np.pad(block1_part, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')
            
            

            affine_blcok1_matrix = affine_address + "/" + '/affine_' + block1_name.split('_')[1].zfill(2) + \
                                   block1_name.split('.')[0].split('_')[2].zfill(2) + '.mat'

            matrix_cell_block1 = sio.loadmat(affine_blcok1_matrix)['affine'][0]
            matrix_cell_block1 = matrix_cell_block1[-pad_block1_part.shape[0] + 1:]
            for i in range(matrix_cell_block1.shape[0] - 1):
                
                matrix_cell_block1[i][0, 2] = matrix_cell_block1[i][0, 2] * affine_scale  
                matrix_cell_block1[i][1, 2] = matrix_cell_block1[i][1, 2] * affine_scale  

                
                error_index = matrix_cell_block1[i][0, 0] * matrix_cell_block1[i][1, 1] - matrix_cell_block1[i][0, 1] * \
                              matrix_cell_block1[i][1, 0]
                if error_index > 1.15 or error_index < 0.85:
                    print('error, change %d' % i)
                    matrix_cell_block1[i] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                

        if block2_name == '0':
            block2_out = np.pad(block2_part, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')
        else:
            pad_block2_part = np.pad(block2_part, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')

            if block1_name == '0':
                
                affine_blcok2_matrix = affine_address + "/" + '/affine_' + block2_name.split('_')[1].zfill(2) + \
                                       block2_name.split('.')[0].split('_')[2].zfill(2) + '.mat'
            else:
                
                affine_blcok2_matrix = affine_address + "/" + '/affine_' + block1_name.split('_')[1].zfill(2) + \
                                       block1_name.split('.')[0].split('_')[2].zfill(2) + '.mat'

            

            matrix_cell_block2 = sio.loadmat(affine_blcok2_matrix)['affine'][0]
            matrix_cell_block2 = matrix_cell_block2[-pad_block2_part.shape[0] + 1:]
            for i in range(matrix_cell_block2.shape[0] - 1):
                
                matrix_cell_block2[i][0, 2] = matrix_cell_block2[i][0, 2] * affine_scale  
                matrix_cell_block2[i][1, 2] = matrix_cell_block2[i][1, 2] * affine_scale  

                
                error_index = matrix_cell_block2[i][0, 0] * matrix_cell_block2[i][1, 1] - matrix_cell_block2[i][0, 1] * \
                              matrix_cell_block2[i][1, 0]
                if error_index > 1.15 or error_index < 0.85:
                    print('error, change %d' % i)
                    matrix_cell_block2[i] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                

        matrix_cell_block = []
        if block2_name == '0' and block1_name == '0':
            del block1_part, block2_part

            return block1_out, block2_out
        else:
            if block2_name == '0':
                matrix_cell_block = matrix_cell_block1
            if block1_name == '0':
                matrix_cell_block = matrix_cell_block2
            if block2_name != '0' and block1_name != '0':
                matrix_cell_block = (matrix_cell_block1 + matrix_cell_block2) / 2

            if block1_name != '0':
                
                block1_out = np.zeros_like(pad_block1_part)
                for i in range(pad_block1_part.shape[0]):
                    im = pad_block1_part[i].T
                    if i == 0:
                        block1_out[i] = im.T
                    else:
                        matrix = compute_matrix(matrix_cell_block, i)
                        block1_out[i] = affine_transform(im, matrix, order=order).T
                
                
                
                
                
                print('finish ' + block1_name + 'line_affine_transform!')

            if block2_name != '0':
                
                block2_out = np.zeros_like(pad_block2_part)
                for i in range(pad_block2_part.shape[0]):
                    im = pad_block2_part[i].T
                    if i == 0:
                        block2_out[i] = im.T
                    else:
                        matrix = compute_matrix(matrix_cell_block, i)  
                        block2_out[i] = affine_transform(im, matrix, order=order).T

                
                
                
                
                
                print('finish ' + block2_name + 'line_affine_transform!')

            del block1_part, block2_part

            return block1_out, block2_out

    elif whether_row == True:

        block1_name_all = FilenameAray[height][hang - 1]
        block2_name_all = FilenameAray[height][hang]

        pad_block1_part = np.pad(block1_part, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')
        pad_block2_part = np.pad(block2_part, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')
        del block1_part, block2_part

        matrix_cell_block1_allfinal = np.zeros(
            (len(FilenameAray[height][hang - 1]), SizeAray[height][hang - 1][0][0] - 1, 3, 3), dtype=np.float)
        block1_whether_zero = np.zeros((len(FilenameAray[height][hang - 1])), dtype=np.float)
        matrix_cell_block2_allfinal = np.zeros(
            (len(FilenameAray[height][hang]), SizeAray[height][hang][0][0] - 1, 3, 3), dtype=np.float)
        block2_whether_zero = np.zeros((len(FilenameAray[height][hang])), dtype=np.float)

        for i in range(len(FilenameAray[height][hang - 1])):

            block1_name = block1_name_all[i].split('.')[0]
            block2_name = block2_name_all[i].split('.')[0]

            if block1_name == '0':
                matrix_cell_block1 = np.zeros((SizeAray[height][hang - 1][0][0] - 1, 3, 3), dtype=np.float)
                matrix_cell_block1_allfinal[i] = matrix_cell_block1
                block1_whether_zero[i] = 0
            else:
                
                
                affine_blcok1_matrix = affine_address + "/" + '/affine_' + block1_name.split('_')[1].zfill(2) + \
                                       block1_name.split('.')[0].split('_')[2].zfill(2) + '.mat'

                matrix_cell_block1 = sio.loadmat(affine_blcok1_matrix)['affine'][0]
                matrix_cell_block1 = matrix_cell_block1[-pad_block1_part.shape[0] + 1:]
                for j in range(matrix_cell_block1.shape[0] - 1):
                    matrix_cell_block1_allfinal[i][j] = matrix_cell_block1[j]
                block1_whether_zero[i] = 1

            if block2_name == '0':
                matrix_cell_block2 = np.zeros((SizeAray[height][hang][0][0] - 1, 3, 3), dtype=np.float)
                matrix_cell_block2_allfinal[i] = matrix_cell_block2
                block2_whether_zero[i] = 0
            else:
                if block1_name == '0':
                    
                    affine_blcok2_matrix = affine_address + "/" + '/affine_' + block2_name.split('_')[1].zfill(2) + \
                                           block2_name.split('.')[0].split('_')[2].zfill(2) + '.mat'
                else:
                    
                    affine_blcok2_matrix = affine_address + "/" + '/affine_' + block1_name.split('_')[1].zfill(2) + \
                                           block1_name.split('.')[0].split('_')[2].zfill(2) + '.mat'
                
                matrix_cell_block2 = sio.loadmat(affine_blcok2_matrix)['affine'][0]
                matrix_cell_block2 = matrix_cell_block2[-pad_block2_part.shape[0] + 1:]
                for j in range(matrix_cell_block2.shape[0] - 1):
                    matrix_cell_block2_allfinal[i][j] = matrix_cell_block2[j]
                block2_whether_zero[i] = 1

        
        matrix_cell_block1 = np.zeros((3, 3), dtype=np.float)
        matrix_cell_block2 = np.zeros((3, 3), dtype=np.float)
        for i in range(len(FilenameAray[height][hang])):
            if block1_whether_zero[i] == 1:
                matrix_cell_block1 = matrix_cell_block1 + matrix_cell_block1_allfinal[i]
            if block2_whether_zero[i] == 1:
                matrix_cell_block2 = matrix_cell_block2 + matrix_cell_block2_allfinal[i]

        matrix_cell_block1 = matrix_cell_block1 / np.sum(block1_whether_zero)
        matrix_cell_block2 = matrix_cell_block2 / np.sum(block2_whether_zero)

        for i in range(matrix_cell_block1.shape[0] - 1):
            
            matrix_cell_block1[i][0, 2] = matrix_cell_block1[i][0, 2] * affine_scale  
            matrix_cell_block1[i][1, 2] = matrix_cell_block1[i][1, 2] * affine_scale  

            
            error_index = matrix_cell_block1[i][0, 0] * matrix_cell_block1[i][1, 1] - matrix_cell_block1[i][0, 1] * \
                          matrix_cell_block1[i][1, 0]
            if error_index > 1.15 or error_index < 0.85:
                print('error, change %d' % i)
                matrix_cell_block1[i] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            

        for i in range(matrix_cell_block2.shape[0] - 1):
            
            matrix_cell_block2[i][0, 2] = matrix_cell_block2[i][0, 2] * affine_scale  
            matrix_cell_block2[i][1, 2] = matrix_cell_block2[i][1, 2] * affine_scale  

            
            error_index = matrix_cell_block2[i][0, 0] * matrix_cell_block2[i][1, 1] - matrix_cell_block2[i][0, 1] * \
                          matrix_cell_block2[i][1, 0]
            if error_index > 1.15 or error_index < 0.85:
                print('error, change %d' % i)
                matrix_cell_block2[i] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            

        matrix_cell_block = (matrix_cell_block1 + matrix_cell_block2) / 2

        
        
        block1_out = np.zeros_like(pad_block1_part)
        for i in range(pad_block1_part.shape[0]):
            im = pad_block1_part[i].T
            if i == 0:
                block1_out[i] = im.T
            else:
                matrix = compute_matrix(matrix_cell_block, i)
                block1_out[i] = affine_transform(im, matrix, order=order).T
        
        
        
        
        
        print('finish ' + 'row_up_affine_transform!')

        
        
        block2_out = np.zeros_like(pad_block2_part)
        for i in range(pad_block2_part.shape[0]):
            im = pad_block2_part[i].T
            if i == 0:
                block2_out[i] = im.T
            else:
                matrix = compute_matrix(matrix_cell_block, i)  
                block2_out[i] = affine_transform(im, matrix, order=order).T
        
        
        
        
        
        print('finish ' + 'row_down_affine_transform!')

        
        

        return block1_out, block2_out


    elif whether_height == True:

        block1_name = FilenameAray[height - 1]
        block2_name = FilenameAray[height]

        return block1_part, block1_part


def get_next_for_per(height, hang, line, whether_z, whether_y, whether_x):
    if whether_x == True:
        next_line = line + 1
        next_hang = hang
        next_height = height
    elif whether_y == True:
        next_line = 1
        next_hang = hang + 1
        next_height = height
    elif whether_z == True:
        next_line = 1
        next_hang = 0
        next_height = height + 1

    return next_height, next_hang, next_line


def update_label(label, height, hang, line, list_old, list_new, direction, cuda_num = 0, width=15):


    if isinstance(width, int):

        return


    else:
        num_x = label.shape[2]
        num_y = label.shape[1]
        num_z = label.shape[0]
        bit_range = label.shape[3]
        label_torch = torch.tensor(label).cuda(cuda_num)

        if direction == 2:

            
            
            
            
            
            
            
            
            
            
            
            

            for num_line in range(0, line + 1):
                for i, j in zip(list_old, list_new):
                    whether = (label_torch[height][hang][num_line] == i)
                    label_torch[height][hang][num_line][whether] = j

        elif direction == 1:
            
            
            
            
            
            

            for num_hang in range(0, hang + 1):
                for num_line in range(0, line):
                    for i, j in zip(list_old, list_new):
                        whether = (label_torch[height][num_hang][num_line] == i)
                        label_torch[height][num_hang][num_line][whether] = j


        elif direction == 0:
            for num_height in range(0, height + 1):
                for num_hang in range(0, hang):
                    for num_line in range(0, line):
                        for i, j in zip(list_old, list_new):
                            whether = (label_torch[num_height][num_hang][num_line] == i)
                            label_torch[num_height][num_hang][num_line][whether] = j

        label = label_torch.cpu().numpy()

        return label




def pair_match(block1_part, block2_part, direction, halo_size):
    direction = int(direction)
    halo_size = int(halo_size)

    
    
    
    
    

    auto_join_pixels = 10000  
    minoverlap_pixels = 2000  
    minoverlap_dual_ratio = 0.5  
    minoverlap_single_ratio = 0.8  

    
    
    
    
    

    
    stacked = np.concatenate([block1_part, block2_part], axis=direction - 1)  

    
    inverse, packed = fastremap.unique(stacked, return_inverse=True)

    
    packed = packed.reshape(stacked.shape)

    if direction == 3:  
        packed_block1 = packed[:, :, :block1_part.shape[2]]
        packed_block2 = packed[:, :, block1_part.shape[2]:]
    elif direction == 2:
        packed_block1 = packed[:, :block1_part.shape[1], :]
        packed_block2 = packed[:, block1_part.shape[1]:, :]
    elif direction == 1:  
        packed_block1 = packed[:block1_part.shape[0], :, :]
        packed_block2 = packed[block1_part.shape[0]:, :, :]

    

    
    
    
    

    
    
    direction = direction - 1

    
    
    

    
    
    packed_overlap1 = packed_block1
    packed_overlap2 = packed_block2
    print("block1", packed_overlap1.shape)
    print("block2", packed_overlap2.shape)

    counter = fast64counter.ValueCountInt64()
    counter.add_values_pair32(packed_overlap1.astype(np.int32).ravel(), packed_overlap2.astype(np.int32).ravel())
    overlap_labels1, overlap_labels2, overlap_areas = counter.get_counts_pair32()

    areacounter = fast64counter.ValueCountInt64()
    areacounter.add_values(packed_overlap1.ravel())
    areacounter.add_values(packed_overlap2.ravel())
    areas = dict(zip(*areacounter.get_counts()))

    to_merge = []
    to_steal = []
    
    for l1, l2, overlap_area in zip(overlap_labels1, overlap_labels2, overlap_areas):

        
        
        

        if inverse[l1] == 0 or inverse[l2] == 0 or l1 == 0 or l2 ==0 :
            continue

        
        

        
        
        
        
        
        

        if ((overlap_area > minoverlap_single_ratio * areas[l1]) or
                (overlap_area > minoverlap_single_ratio * areas[l2]) or
                ((overlap_area > minoverlap_dual_ratio * areas[l1]) and
                 (overlap_area > minoverlap_dual_ratio * areas[l2]))):

            
            
            

            if inverse[l1] != inverse[l2]:
                
                to_merge.append((inverse[l1], inverse[l2]))
            
        else:
            
            to_steal.append((overlap_area, l1, l2))

    merge_map = list((sorted(s, reverse=True)) for s in to_merge)
    merge_map = [value for index, value in
                 sorted(enumerate(merge_map), key=lambda merge_map: merge_map[1], reverse=True)]
    
    
    

    
    
    
    
    
    

    
    list_old = []
    list_new = []
    G = nx.Graph()
    G.add_edges_from(merge_map)
    
    for c in nx.connected_components(G):
        
        nodeSet = G.subgraph(c).nodes()

        target = min(nodeSet)
        for label_merge in nodeSet:
            
            
            
            

            if label_merge != target:
                list_old.append(label_merge)
                list_new.append(target)
    

    

    
    
    
    

    

    
    return list_old, list_new


def merge_tif_bigdata_not_Regular_stack_merge_z(stack_path, save_path, raw_path, text_path, overloap_x_pixel, overloap_y_pixel, overloap_z_pixel, bite,
                                                cuda_num, width, start_num, end_num):

    

    load_path = stack_path
    
    save_path = save_path
    raw_path = raw_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)



    

    
    


    all_stack = os.listdir(load_path)
    all_stack.sort(key=lambda x: int(x[5:]))

    all_stack = all_stack[start_num:end_num]
    print(all_stack)

    total_z = int(len(all_stack))  

    max_y = 15
    max_x = 15

    FilenameAray_xy = []
    FilenameAray = []
    for j in range(total_z):
        FilenameAray_xy = []
        for i in range(max_y):
            zeroArray = ['0' for i in range(max_x)]
            FilenameAray_xy.append(zeroArray)
        FilenameAray.append(FilenameAray_xy)

    del FilenameAray_xy

    SizeAray = []
    SizeAray_xy = []
    for j in range(total_z):
        SizeAray_xy = []
        for i in range(max_y):
            zeroArray = [[0, 0, 0] for i in range(max_x)]
            SizeAray_xy.append(zeroArray)
        SizeAray.append(SizeAray_xy)
    del SizeAray_xy

    num_x_list = []
    num_y_list = []

    for stack_num in range(total_z):
        sub_data_path = os.path.join(load_path, all_stack[stack_num])
        sub_tifs = os.listdir(sub_data_path)

        total = int(len(sub_tifs))  

        sub_tifs.sort(
            key=lambda x: int(x.split('-')[0] * 100 + x.split('_')[1] * 10 + x.split('.')[0].split('_')[2] * 1))
        
        sorted_file = sub_tifs

        
        
        row_all = np.zeros(total)
        line_all = np.zeros(total)

        
        
        for i, name in enumerate(sorted_file):
            row_all[i] = int(name.split('_')[1])
            
        for i, name in enumerate(sorted_file):
            line_all[i] = int(name.split('.')[0].split('_')[2])
            

        
        number_y = np.unique(row_all)
        number_x = np.unique(line_all)

        
        num_y = len(number_y)
        num_x = len(number_x)

        num_x_list.append(num_x)
        num_y_list.append(num_y)

        
        all_y = number_y.tolist()
        all_x = number_x.tolist()

        del row_all, line_all, number_y, number_x

        for name in (sorted_file):
            name_row = int(name.split('_')[1])
            name_line = int(name.split('.')[0].split('_')[2])
            
            
            
            

            
            y = all_y.index(name_row)
            x = all_x.index(name_line)

            
            
            FilenameAray[stack_num][y][x] = name

        
        size_z, size_y, size_x = get_one_block_size(sub_data_path, whether_loadpath=True)
        print('block size_z:', size_z)
        print('block size_y:', size_y)
        print('block size_x:', size_x)
        

        

        size_z = 0
        size_y = 0
        size_x = 0

        
        for hang in range(0, num_y):
            for line in range(0, num_x):
                if FilenameAray[stack_num][hang][line] == '0':
                    pass
                else:
                    try:
                        
                        size_z, size_y, size_x = get_one_block_size(sub_data_path + '/' + FilenameAray[stack_num][hang][line], whether_loadpath=False)

                        SizeAray[stack_num][hang][line][0] = size_z
                        SizeAray[stack_num][hang][line][1] = size_y
                        SizeAray[stack_num][hang][line][2] = size_x
                    except:
                        print('height: ', stack_num, 'hang: ', hang,'line: ', line,  '  ', FilenameAray[stack_num][hang][line], ' is wrong!')
        
        
        for hang in range(0, num_y):
            for line in range(0, num_x):
                if FilenameAray[stack_num][hang][line] != '0':
                    pass
                else:
                    for i in range(0, num_x):
                        if FilenameAray[stack_num][hang][i] != '0':
                            SizeAray[stack_num][hang][line][1] = SizeAray[stack_num][hang][i][1]
                            SizeAray[stack_num][hang][line][0] = SizeAray[stack_num][hang][i][0]
                            break
                    for j in range(0, num_y):
                        if FilenameAray[stack_num][j][line] != '0':
                            SizeAray[stack_num][hang][line][2] = SizeAray[stack_num][j][line][2]
                            break
        

    print(len(FilenameAray))
    print(len(SizeAray))

    height = 0
    hang = 0
    line = 1
    print('initialiaztion completed!')

    max_height_last = 0
    for height in range(0, total_z): 

        print('now start from ', all_stack[height])

        if height == 0:
            continue
        
        

        

        
        try:
            if not os.path.exists(save_path + '/' + all_stack[height-1] + '/' + all_stack[height-1] + '/' + "merge2.dat"):
                inital_z, inital_y, inital_x, label1 = load_binary_dat_break_down(save_path + '/' + all_stack[height - 1] + '/' + all_stack[height-1] + '/' + "merge.dat")  
                print('load ' , save_path + '/' + all_stack[height-1] + '/' + all_stack[height-1] + '/' + "merge.dat" , ' is successful')
            else:
                inital_z, inital_y, inital_x, label1 = load_binary_dat_break_down(save_path + '/' + all_stack[height - 1] + '/' + all_stack[height-1] + '/' + "merge2.dat")  
                print('load ', save_path + '/' + all_stack[height - 1] + '/' + all_stack[height - 1] + '/' + "merge2.dat", ' is successful')
        except:
            print(save_path + '/' + all_stack[height - 1] + '/' + all_stack[height-1] + " merge.dat is not existing!")

        
        

        
        try:
            if not os.path.exists(save_path + '/' + all_stack[height] + '/' + all_stack[height] + '/' + "merge2.dat"):
                inital_z, inital_y, inital_x, label2 = load_binary_dat_break_down(save_path + '/' + all_stack[height] + '/' + all_stack[height] + '/' + "merge.dat")  
                print('load ', save_path + '/' + all_stack[height] + '/' + all_stack[height] + '/' + "merge.dat", ' is successful')
            else:
                inital_z, inital_y, inital_x, label2 = load_binary_dat_break_down(save_path + '/' + all_stack[height] + '/' + all_stack[height] + '/' + "merge2.dat")  
                print('load ', save_path + '/' + all_stack[height] + '/' + all_stack[height] + '/' + "merge2.dat", ' is successful')

        except:
            print(save_path + '/' + all_stack[height] + '/' + all_stack[height] + " merge.dat is not existing!")

        
        
        
        
        
        
        
        
        
        
        

        num_x_1 = num_x_list[height-1]

        num_y_1 = num_y_list[height-1]

        start = time.process_time()

        block1_part, max_block1_part_all = read_chongdie_area_all(FilenameAray, SizeAray, load_path + all_stack[height-1] + '/', num_x_1, num_y_1, True, 
                                                                  height , label1, overloap_x_pixel, overloap_y_pixel,
                                                                  halo_size=overloap_z_pixel)
        end = time.process_time()
        print('read_chongdie_area block1 spend time is ', end - start)

        num_x_2 = num_x_list[height]

        num_y_2 = num_y_list[height]

        start = time.process_time()

        block2_part, max_block2_part_all = read_chongdie_area_all(FilenameAray, SizeAray, load_path + all_stack[height] + '/', num_x_2, num_y_2, False, 
                                                                  height,  label2, overloap_x_pixel, overloap_y_pixel,
                                                                  halo_size=overloap_z_pixel)

        end = time.process_time()
        print('read_chongdie_area block2 spend time is ', end - start)


        
        if max_height_last <= max_block1_part_all:
            max_height_last = max_block1_part_all  
        elif max_block1_part_all == 0:
            print('block1_part z is a blank block')
        else:
            print('block1_part z get max_block1_part_all error')
            
        


        row_new, row_next, label2 = make_label(block1_part, block2_part, max_height_last, max_block2_part_all, label2,     
                                              0, hang=num_y_list[height], line=num_x_list[height],     
                                              direction=1)

        

        text_path_1 = text_path + all_stack[height - 1] + '/' + 'mosaic2fiji.txt'
        text_path_2 = text_path + all_stack[height] + '/' + 'mosaic2fiji.txt'


        f = open(text_path_1)  
        
        line = f.readline()
        print(line)
        
        
        yoff_1 = np.int32(np.int32(line.split(',')[2]))
        xoff_1 = np.int32(np.int32(line.split(',')[1]))

        
        first_name_real_image = FilenameAray[height - 1][0][0]
        real_y = np.int32(first_name_real_image.split('_')[-2])
        suoyin_y =np.int32(line.split('.')[0].split('_')[-2])
        yoff_1 = yoff_1 + (real_y - suoyin_y) * (SizeAray[height - 1][0][0][1] - overloap_y_pixel)

        real_x = np.int32(first_name_real_image.split('_')[-1])
        suoyin_x = np.int32(line.split('.')[0].split('_')[-1])
        xoff_1 = xoff_1 + (real_x - suoyin_x) * (SizeAray[height - 1][0][0][2] - overloap_x_pixel)
        

        f.close()


        f = open(text_path_2)  
        
        line = f.readline()
        print(line)
        
        
        yoff_2 = np.int32(np.int32(line.split(',')[2]))
        xoff_2 = np.int32(np.int32(line.split(',')[1]))

        
        first_name_real_image = FilenameAray[height][0][0]
        real_y = np.int32(first_name_real_image.split('_')[-2])
        suoyin_y = np.int32(line.split('.')[0].split('_')[-2])
        yoff_2 = yoff_2 + (real_y - suoyin_y) * (SizeAray[height][0][0][1] - overloap_y_pixel)

        real_x = np.int32(first_name_real_image.split('_')[-1])
        suoyin_x = np.int32(line.split('.')[0].split('_')[-1])
        xoff_2 = xoff_2 + (real_x - suoyin_x) * (SizeAray[height][0][0][2] - overloap_x_pixel)
        

        f.close()

        distence_x = np.int32(np.abs(xoff_2 - xoff_1))
        distence_y = np.int32(np.abs(yoff_2 - yoff_1))


        
        length_1_x = 0
        length_1_x = np.int32(length_1_x)
        length_1_y = 0
        length_1_y = np.int32(length_1_y)
        length_2_x = 0
        length_2_x = np.int32(length_2_x)
        length_2_y = 0
        length_2_y = np.int32(length_2_y)

        for i in range(num_x_1):
            if i == 0:
                length_1_x = length_1_x + np.int32(SizeAray[height-1][0][i][2])
            else:
                length_1_x = length_1_x + np.int32(SizeAray[height - 1][0][i][2]) - overloap_x_pixel
        for i in range(num_y_1):
            if i == 0:
                length_1_y = length_1_y + np.int32(SizeAray[height-1][i][0][1])
            else:
                length_1_y = length_1_y + np.int32(SizeAray[height - 1][i][0][1]) - overloap_y_pixel
        for i in range(num_x_2):
            if i == 0:
                length_2_x = length_2_x + np.int32(SizeAray[height][0][i][2])
            else:
                length_2_x = length_2_x + np.int32(SizeAray[height][0][i][2]) - overloap_x_pixel
        for i in range(num_y_2):
            if i == 0:
                length_2_y = length_2_y + np.int32(SizeAray[height][i][0][1])
            else:
                length_2_y = length_2_y + np.int32(SizeAray[height][i][0][1]) - overloap_y_pixel

        cha_x = xoff_2 - xoff_1 + length_2_x - length_1_x
        cha_y = yoff_2 - yoff_1 + length_2_y - length_1_y

        abs_cha_x = np.int32(np.abs(cha_x))
        abs_cha_y = np.int32(np.abs(cha_y))



        if xoff_2 >= xoff_1 and yoff_2 >= yoff_1:

            if cha_x >= 0 and cha_y >= 0:

                
                

                coordinate_1_x_start = distence_x
                coordinate_1_x_end = length_1_x - 1
                coordinate_1_y_start = distence_y
                coordinate_1_y_end = length_1_y - 1
                coordinate_2_x_start = 0
                coordinate_2_x_end = length_2_x - abs_cha_x - 1
                coordinate_2_y_start = 0
                coordinate_2_y_end = length_2_y - abs_cha_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]


            elif cha_x >= 0 and cha_y <= 0:
                
                

                coordinate_1_x_start = distence_x
                coordinate_1_x_end = length_1_x - 1
                coordinate_1_y_start = distence_y
                coordinate_1_y_end = length_1_y - abs_cha_y - 1
                coordinate_2_x_start = 0
                coordinate_2_x_end = length_2_x - abs_cha_x - 1
                coordinate_2_y_start = 0
                coordinate_2_y_end = length_2_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]
            elif cha_x <= 0 and cha_y >= 0:
                
                

                coordinate_1_x_start = distence_x
                coordinate_1_x_end = length_1_x - abs_cha_x - 1
                coordinate_1_y_start = distence_y
                coordinate_1_y_end = length_1_y - 1
                coordinate_2_x_start = 0
                coordinate_2_x_end = length_2_x - 1
                coordinate_2_y_start = 0
                coordinate_2_y_end = length_2_y - abs_cha_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]
            elif cha_x <= 0 and cha_y <= 0:
                
                
                coordinate_1_x_start = distence_x
                coordinate_1_x_end = length_1_x - abs_cha_x - 1
                coordinate_1_y_start = distence_y
                coordinate_1_y_end = length_1_y - abs_cha_y - 1
                coordinate_2_x_start = 0
                coordinate_2_x_end = length_2_x - 1
                coordinate_2_y_start = 0
                coordinate_2_y_end = length_2_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]

        elif xoff_2 >= xoff_1 and yoff_2 <= yoff_1:
            if cha_x >= 0 and cha_y >= 0:
                
                

                coordinate_1_x_start = distence_x
                coordinate_1_x_end = length_1_x - 1
                coordinate_1_y_start = 0
                coordinate_1_y_end = length_1_y - 1
                coordinate_2_x_start = 0
                coordinate_2_x_end = length_2_x - abs_cha_x - 1
                coordinate_2_y_start = distence_y
                coordinate_2_y_end = length_2_y - abs_cha_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]
            elif cha_x >= 0 and cha_y <= 0:
                
                

                coordinate_1_x_start = distence_x
                coordinate_1_x_end = length_1_x - 1
                coordinate_1_y_start = 0
                coordinate_1_y_end = length_1_y - abs_cha_y - 1
                coordinate_2_x_start = 0
                coordinate_2_x_end = length_2_x - abs_cha_x - 1
                coordinate_2_y_start = distence_y
                coordinate_2_y_end = length_2_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]
            elif cha_x <= 0 and cha_y >= 0:
                
                

                coordinate_1_x_start = distence_x
                coordinate_1_x_end = length_1_x - abs_cha_x - 1
                coordinate_1_y_start = 0
                coordinate_1_y_end = length_1_y - 1
                coordinate_2_x_start = 0
                coordinate_2_x_end = length_2_x - 1
                coordinate_2_y_start = distence_y
                coordinate_2_y_end = length_2_y - abs_cha_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]
            elif cha_x <= 0 and cha_y <= 0:
                
                

                coordinate_1_x_start = distence_x
                coordinate_1_x_end = length_1_x - abs_cha_x - 1
                coordinate_1_y_start = 0
                coordinate_1_y_end = length_1_y - abs_cha_y - 1
                coordinate_2_x_start = 0
                coordinate_2_x_end = length_2_x - 1
                coordinate_2_y_start = distence_y
                coordinate_2_y_end = length_2_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]
        elif xoff_2 <= xoff_1 and yoff_2 >= yoff_1:
            if cha_x >= 0 and cha_y >= 0:
                
                

                coordinate_1_x_start = 0
                coordinate_1_x_end = length_1_x - 1
                coordinate_1_y_start = distence_y
                coordinate_1_y_end = length_1_y - 1
                coordinate_2_x_start = distence_x
                coordinate_2_x_end = length_2_x - abs_cha_x - 1
                coordinate_2_y_start = 0
                coordinate_2_y_end = length_2_y - abs_cha_y -1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]
            elif cha_x >= 0 and cha_y <= 0:
                
                

                coordinate_1_x_start = 0
                coordinate_1_x_end = length_1_x - 1
                coordinate_1_y_start = distence_y
                coordinate_1_y_end = length_1_y - abs_cha_y - 1
                coordinate_2_x_start = distence_x
                coordinate_2_x_end = length_2_x - abs_cha_x - 1
                coordinate_2_y_start = 0
                coordinate_2_y_end = length_2_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]
            elif cha_x <= 0 and cha_y >= 0:
                
                

                coordinate_1_x_start = 0
                coordinate_1_x_end = length_1_x - abs_cha_x - 1
                coordinate_1_y_start = distence_y
                coordinate_1_y_end = length_1_y - 1
                coordinate_2_x_start = distence_x
                coordinate_2_x_end = length_2_x - 1
                coordinate_2_y_start = 0
                coordinate_2_y_end = length_2_y - abs_cha_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]
            elif cha_x <= 0 and cha_y <= 0:
                
                

                coordinate_1_x_start = 0
                coordinate_1_x_end = length_1_x - abs_cha_x - 1
                coordinate_1_y_start = distence_y
                coordinate_1_y_end = length_1_y - abs_cha_y - 1
                coordinate_2_x_start = distence_x
                coordinate_2_x_end = length_2_x - 1
                coordinate_2_y_start = 0
                coordinate_2_y_end = length_2_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]
        else:
            if cha_x >= 0 and cha_y >= 0:
                
                

                coordinate_1_x_start = 0
                coordinate_1_x_end = length_1_x - 1
                coordinate_1_y_start = 0
                coordinate_1_y_end = length_1_y - 1
                coordinate_2_x_start = distence_x
                coordinate_2_x_end = length_2_x - abs_cha_x - 1
                coordinate_2_y_start = distence_y
                coordinate_2_y_end = length_2_y - abs_cha_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]
            elif cha_x >= 0 and cha_y <= 0:
                
                

                coordinate_1_x_start = 0
                coordinate_1_x_end = length_1_x - 1
                coordinate_1_y_start = 0
                coordinate_1_y_end = length_1_y - abs_cha_y - 1
                coordinate_2_x_start = distence_x
                coordinate_2_x_end = length_2_x - abs_cha_x - 1
                coordinate_2_y_start = distence_y
                coordinate_2_y_end = length_2_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]
            elif cha_x <= 0 and cha_y >= 0:
                
                

                coordinate_1_x_start = 0
                coordinate_1_x_end = length_1_x - abs_cha_x - 1
                coordinate_1_y_start = 0
                coordinate_1_y_end = length_1_y - 1
                coordinate_2_x_start = distence_x
                coordinate_2_x_end = length_2_x - 1
                coordinate_2_y_start = distence_y
                coordinate_2_y_end = length_2_y - abs_cha_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]
            elif cha_x <= 0 and cha_y <= 0:
                
                

                coordinate_1_x_start = 0
                coordinate_1_x_end = length_1_x - abs_cha_x - 1
                coordinate_1_y_start = 0
                coordinate_1_y_end = length_1_y - abs_cha_y - 1
                coordinate_2_x_start = distence_x
                coordinate_2_x_end = length_2_x - 1
                coordinate_2_y_start = distence_y
                coordinate_2_y_end = length_2_y - 1

                row_new = row_new[:, coordinate_1_y_start:coordinate_1_y_end, coordinate_1_x_start:coordinate_1_x_end]
                row_next = row_next[:, coordinate_2_y_start:coordinate_2_y_end, coordinate_2_x_start:coordinate_2_x_end]



        
        
        
        
        
        
        
        
        
        
        
        
        

        assert row_new.shape == row_next.shape


        
        

        

        
        

        
        

        
        

        start = time.process_time()
        list_old, list_new = pair_match(row_new, row_next,  direction=1, halo_size=overloap_z_pixel)

        end = time.process_time()
        print('pair_match spend time is ', end - start)


        start_height = height - width
        
        

        if start_height <= 0: 
            start_height = 0

        print('update is ready for ', all_stack[height], ' to ', all_stack[start_height])

        for i in range(height, start_height-1, -1):
            a = 1

            

            if i == height:
                start = time.process_time()
                label = update_label(label2, 0, num_y_list[i], num_x_list[i], list_old, list_new, direction=0, cuda_num=cuda_num, width='all')  
                end = time.process_time()
                print(all_stack[i], ' update_label spend time is ', end - start)
            elif i == height - 1:
                start = time.process_time()
                label = update_label(label1, 0, num_y_list[i], num_x_list[i], list_old, list_new, direction=0, cuda_num=cuda_num, width='all')  
                end = time.process_time()
                print(all_stack[i], ' update_label spend time is ', end - start)
            else:

                if not os.path.exists(save_path + '/' + all_stack[i] + '/' + all_stack[i] + '/' + "merge2.dat"):
                    print('line 1877 error, where is merge2.dat ???')
                    return
                else:
                    inital_z, inital_y, inital_x, label = load_binary_dat_break_down(save_path + '/' + all_stack[i] + '/' + all_stack[i] + '/' + "merge2.dat")  
                    print('in update! load ', save_path + '/' + all_stack[i] + '/' + all_stack[i] + '/' + "merge2.dat", ' is successful')

                start = time.process_time()
                label = update_label(label, 0, num_y_list[i], num_x_list[i], list_old, list_new, direction=0, cuda_num=cuda_num, width='all')  
                end = time.process_time()
                print(all_stack[i], ' update_label spend time is ', end - start)

            save_2_path = save_path + '/' + all_stack[i] + '/' + all_stack[i] 

            save_name = 'merge2.dat'

            

            start = time.process_time()
            save_binary_dat(label, save_2_path, bite, 1, num_y_list[i], num_x_list[i], overloap_z_pixel, overloap_y_pixel,         
                            overloap_x_pixel, SizeAray[i][num_y_list[i]-1][num_x_list[i]-1][0], SizeAray[i][num_y_list[i]-1][num_x_list[i]-1][1], SizeAray[i][num_y_list[i]-1][num_x_list[i]-1][2],
                            height, num_y_list[i]-1, num_x_list[i]-1, save_name)  

            end = time.process_time()
            print(save_2_path + '/'+ save_name +' spend time is ', end - start)
            del label

        del label1,label2


        
        
        
        

    
        
        
        
    
    
    

    print('z merge is all finish!')
    print('begin write!')

    for height in range(0, total_z):

        sub_data_path = os.path.join(load_path, all_stack[height])
        sub_tifs = os.listdir(sub_data_path)

        total = int(len(sub_tifs))  

        sub_tifs.sort(
            key=lambda x: int(x.split('-')[0] * 100 + x.split('_')[1] * 10 + x.split('.')[0].split('_')[2] * 1))
        
        sorted_file = sub_tifs

        inital_z, inital_y, inital_x, label = load_binary_dat_break_down(
            save_path + '/' + all_stack[height] + '/' + all_stack[height] + '/' + "merge2.dat")  
        print('in writing! load ', save_path + '/' + all_stack[height] + '/' + all_stack[height] + '/' + "merge2.dat", ' is successful')


        


def make_big_label(raw_path, load_path, sorted_file, save_path, label, num_x, num_y, num_z, overloap_x_pixel,
                   overloap_y_pixel, overloap_z_pixel):
    print('begin make_big_label')

    
    
    

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    
    

    
    
    
    
    
    
    
    
    

    ROWs_raw = []
    HEIGHTs_raw = []
    ROWs_label = []
    HEIGHTs_label = []
    for height in range(0, num_z):
        print('height: ' + str(height))
        for hang in range(0, num_y):
            print('hang: ' + str(hang))
            row_first_name_label = load_path + '/' + str(sorted_file[hang * num_x + height * num_x * num_y])
            

            
            

            
            

            
            row_new_label = read_32_tif_seqqence(row_first_name_label)
            row_new_label = label_update(row_new_label, label, height, hang, 0)
            print('label update: ' + row_first_name_label)
            print('label update shape:', row_new_label.shape)

            
            

            for line in range(1, num_x):
                print('line: ' + str(line))
                row_next_name_label = load_path + '/' + str(sorted_file[hang * num_x + line + height * num_x * num_y])
                

                
                
                
                

                
                
                
                
                

                
                row_next_label = read_32_tif_seqqence(row_next_name_label)
                row_next_label = label_update(row_next_label, label, height, hang, line)
                print('label update: ' + row_next_name_label)
                print('label update shape:', row_next_label.shape)

                
                

                
                
                
                

                

                
                row_new_label, row_next_label = overlap_fusing(row_new_label, row_next_label, direction=3,
                                                               halo_size=overloap_x_pixel,
                                                               whether_label=True)

                row_new_label = np.concatenate([row_new_label, row_next_label[:, :, overloap_x_pixel:]], axis=2)

            if hang == 0:
                
                ROWs_label = row_new_label
            else:
                
                
                

                

                
                ROWs_label, row_new_label = overlap_fusing(ROWs_label, row_new_label, direction=2,
                                                           halo_size=overloap_y_pixel,
                                                           whether_label=True)

                ROWs_label = np.concatenate([ROWs_label, row_new_label[:, overloap_y_pixel:, :]], axis=1)

        if height == 0:
            
            HEIGHTs_label = ROWs_label
        else:
            
            
            

            

            
            HEIGHTs_label, ROWs_label = overlap_fusing(HEIGHTs_label, ROWs_label, direction=1,
                                                       halo_size=overloap_z_pixel,
                                                       whether_label=True)

            HEIGHTs_label = np.concatenate([HEIGHTs_label, ROWs_label[overloap_z_pixel:, :, :]], axis=0)

    HEIGHTs_label = np.uint32(HEIGHTs_label)

    save_raw_path = save_path + '/' + 'big_raw'
    save_label_path = save_path + '/' + 'big_label1'
    save_color_path = save_path + '/' + 'colorful'

    whether_save_raw = False
    whether_save_label = True
    whether_save_colorful = False


    if whether_save_raw == True:
        if not os.path.exists(save_raw_path):
            os.mkdir(save_raw_path)
        names = os.listdir(save_raw_path)
        total = int(len(names))  
        if total < HEIGHTs_raw.shape[0]:
            print('begin write raw')
            for i in range(0, HEIGHTs_raw.shape[0]):
                imageio.imwrite(save_raw_path + '/' + str(i + 1).zfill(4) + '.tif', HEIGHTs_raw[i])
        else:
            print(save_raw_path, ' has already writed raw')

    if whether_save_label == True:
        if not os.path.exists(save_label_path):
            os.mkdir(save_label_path)
        names = os.listdir(save_label_path)
        total = int(len(names))  
        if total <= HEIGHTs_label.shape[0]:
            print('begin write label')
            for i in range(0, HEIGHTs_label.shape[0]):
                
                img2d = Image.fromarray(HEIGHTs_label[i], 'RGBA')
                img2d.save(os.path.join(save_label_path, str(i + 1).zfill(4) + '.png'))
        else:
            print(save_label_path, ' has already writed label')

    if whether_save_colorful == True:
        if not os.path.exists(save_color_path):
            os.mkdir(save_color_path)
        names = os.listdir(save_color_path)
        total = int(len(names))  
        if total < HEIGHTs_label.shape[0]:
            f = h5py.File(r'./colorMap.hdf5', 'r')
            color_map = f['idColorMap'][:]
            print('begin write colorful')

            for i in range(0, HEIGHTs_label.shape[0]):
                im_syn2D = overlay(HEIGHTs_label, i, color_map, HEIGHTs_raw)
                imageio.imwrite(save_color_path + '/' + str(i + 1).zfill(4) + '.tif', im_syn2D)
        else:
            print(save_color_path, ' has already writed colorful')


def overlay(syn2D, i, color_map, raw):
    img1 = color_map[np.mod(syn2D[i], color_map.shape[0])]
    img2 = np.stack([raw[i], raw[i], raw[i]], axis=2)
    img2[img1[:, :, 0] != 0] = (img1 * 0.7 + img2 * 0.3)[img1[:, :, 0] != 0]
    return img2


def show_colorful_label():
    raw_path = '/home/liujz/liujz/2022_zbf_thire/zbfStackResults_merge/stack540_4_raw/'
    

    raw = read_32_tif_seqqence(raw_path)
    raw = np.uint8(raw)

    print('over read raw')

    label_path1 = '/home/liujz/liujz/2022_zbf_thire/zbfStackResults_merge/stack540_big/stack540/'
    
    label = read_32_tif_seqqence(label_path1)

    print('over read label')

    

    

    f = h5py.File(r'./colorMap.hdf5', 'r')
    color_map = f['idColorMap'][:]

    def overlay(syn2D, i, color_map, raw):
        img1 = color_map[np.mod(syn2D[i], color_map.shape[0])]
        img2 = np.stack([raw[i], raw[i], raw[i]], axis=2)
        img2[img1[:, :, 0] != 0] = (img1 * 0.7 + img2 * 0.3)[img1[:, :, 0] != 0]
        return img2

    save_path = '/home/liujz/liujz/2022_zbf_thire/zbfStackResults_merge/stack540_colorful/'
    

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    print('begin write')

    for i in range(0, label.shape[0]):
        im_syn2D = overlay(label, i, color_map, raw)
        imageio.imwrite(save_path + '/' + str(i).zfill(3) + '.tif', im_syn2D)


def load_binary_dat_save_tif(dat_path, raw_label_path, save_path):
    _, num_z, num_y, num_x, label = load_binary_dat(dat_path)

    num_z = int(num_z[0])
    num_y = int(num_y[0])
    num_x = int(num_x[0])

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    sub_data_path = os.path.join(raw_label_path)
    sub_tifs = os.listdir(sub_data_path)
    total = int(len(sub_tifs))  
    sub_tifs.sort(key=lambda x: int(x.split('-')[0]) * 100 + int(x.split('_')[1]) * 10 + int(x.split('_')[2]))

    sorted_file = sub_tifs
    
    height_all = np.zeros(total)
    row_all = np.zeros(total)
    line_all = np.zeros(total)
    for i, name in enumerate(sorted_file):
        height_all[i] = int(name.split('-')[0])
    for i, name in enumerate(sorted_file):
        row_all[i] = int(name.split('_')[1])
    for i, name in enumerate(sorted_file):
        line_all[i] = int(name.split('_')[2])

    number_z = np.unique(height_all)
    number_y = np.unique(row_all)
    number_x = np.unique(line_all)

    cal_num_z = len(number_z)
    cal_num_y = len(number_y)
    cal_num_x = len(number_x)

    assert cal_num_z == num_z
    assert cal_num_y == num_y
    assert cal_num_x == num_x

    all_z = number_z.tolist()
    all_y = number_y.tolist()
    all_x = number_x.tolist()
    FilenameAray_xy = []
    FilenameAray = []
    for j in range(num_z):
        FilenameAray_xy = []
        for i in range(num_y):
            zeroArray = ['0' for i in range(num_x)]
            FilenameAray_xy.append(zeroArray)
        FilenameAray.append(FilenameAray_xy)

    min_y = int(np.min(row_all))
    
    min_x = int(np.min(line_all))
    

    del row_all, line_all, FilenameAray_xy, number_y, number_x

    for name in (sorted_file):
        name_row = int(name.split('_')[1])
        name_line = int(name.split('_')[2])
        name_height = int(name.split('-')[0])
        z = all_z.index(name_height)
        y = all_y.index(name_row)
        x = all_x.index(name_line)
        FilenameAray[z][y][x] = name

    img_upadte_all(FilenameAray, raw_label_path, sorted_file, save_path, label, num_x, num_y, num_z)


def save_software_every_binary_dat(dat_path, raw_label_path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(dat_path, 'rb') as f:
        bite = struct.unpack('i', f.read(4))  
        num_z = struct.unpack('i', f.read(4))  
        num_y = struct.unpack('i', f.read(4))  
        num_x = struct.unpack('i', f.read(4))  
        overloap_z_pixel = struct.unpack('i', f.read(4))  
        overloap_y_pixel = struct.unpack('i', f.read(4))  
        overloap_x_pixel = struct.unpack('i', f.read(4))  
        size_z = struct.unpack('i', f.read(4))  
        size_y = struct.unpack('i', f.read(4))  
        size_x = struct.unpack('i', f.read(4))  
        height = struct.unpack('i', f.read(4))  
        hang = struct.unpack('i', f.read(4))  
        line = struct.unpack('i', f.read(4))  

        label = np.zeros((int(num_z[0]), int(num_y[0]), int(num_x[0]), 2 ** int(bite[0])), dtype=np.int32)

        arr_read = struct.unpack(
            '{}i'.format((int(num_x[0]) * int(num_y[0]) * int(num_z[0])) * (2 ** int(bite[0]))),
            f.read())  
        arr_read = np.array(list(arr_read), dtype=np.int32)
        arr_read = arr_read.reshape(label.shape)

    num_z = int(num_z[0])
    num_y = int(num_y[0])
    num_x = int(num_x[0])

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    sub_data_path = os.path.join(raw_label_path)

    sub_tifs = os.listdir(sub_data_path)
    total = int(len(sub_tifs))  
    
    sub_tifs.sort(key=lambda x: int(x.split('-')[0] * 100 + x.split('_')[1] * 10 + x.split('.')[0].split('_')[2] * 1))

    sorted_file = sub_tifs
    
    height_all = np.zeros(total)
    row_all = np.zeros(total)
    line_all = np.zeros(total)
    for i, name in enumerate(sorted_file):
        height_all[i] = int(name.split('-')[0])
    for i, name in enumerate(sorted_file):
        row_all[i] = int(name.split('_')[1])
    for i, name in enumerate(sorted_file):
        line_all[i] = int(name.split('.')[0].split('_')[2])

    number_z = np.unique(height_all)
    number_y = np.unique(row_all)
    number_x = np.unique(line_all)

    cal_num_z = len(number_z)
    cal_num_y = len(number_y)
    cal_num_x = len(number_x)

    assert cal_num_z == num_z
    assert cal_num_y == num_y
    assert cal_num_x == num_x

    all_z = number_z.tolist()
    all_y = number_y.tolist()
    all_x = number_x.tolist()
    FilenameAray_xy = []
    FilenameAray = []
    for j in range(num_z):
        FilenameAray_xy = []
        for i in range(num_y):
            zeroArray = ['0' for i in range(num_x)]
            FilenameAray_xy.append(zeroArray)
        FilenameAray.append(FilenameAray_xy)

    min_y = int(np.min(row_all))
    
    min_x = int(np.min(line_all))
    

    del row_all, line_all, FilenameAray_xy, number_y, number_x

    for name in (sorted_file):
        name = name.split('.')[0]

        name_row = int(name.split('_')[1])
        name_line = int(name.split('.')[0].split('_')[2])
        name_height = int(name.split('-')[0])
        z = all_z.index(name_height)
        y = all_y.index(name_row)  
        x = all_x.index(name_line)  
        FilenameAray[z][y][x] = name

    print('begin write')

    with open(save_path + '/' + 'head' + '.dat',
              "wb") as outfile:  
        num_bite_0 = struct.pack('i', bite[0])  
        outfile.write(num_bite_0)

        num_z_1 = struct.pack('i', num_z)  
        outfile.write(num_z_1)

        num_y_2 = struct.pack('i', num_y)  
        outfile.write(num_y_2)

        num_x_3 = struct.pack('i', num_x)  
        outfile.write(num_x_3)

        overloap_z_pixel_4 = struct.pack('i',
                                         overloap_z_pixel[0])  
        outfile.write(overloap_z_pixel_4)

        overloap_y_pixel_5 = struct.pack('i',
                                         overloap_y_pixel[0])  
        outfile.write(overloap_y_pixel_5)

        overloap_x_pixel_6 = struct.pack('i',
                                         overloap_x_pixel[0])  
        outfile.write(overloap_x_pixel_6)

        
        size_z_7 = struct.pack('i', size_z[0])  
        outfile.write(size_z_7)

        size_y_8 = struct.pack('i', size_y[0])  
        outfile.write(size_y_8)

        size_x_9 = struct.pack('i', size_x[0])  
        outfile.write(size_x_9)

        height_10 = struct.pack('i', height[0])  
        outfile.write(height_10)

        hang_11 = struct.pack('i', hang[0])  
        outfile.write(hang_11)

        line_12 = struct.pack('i', line[0])  
        outfile.write(line_12)

    for file in sorted_file:

        
        

        file = file.split('.')[0]

        with open(save_path + '/' + file + ".dat",
                  "wb") as outfile:  

            Whether_found = False
            for i in range(0, num_z):
                if Whether_found == True:
                    break
                for j in range(0, num_y):
                    if Whether_found == True:
                        break
                    for k in range(0, num_x):
                        if FilenameAray[i][j][k] == file:
                            
                            index_z = i
                            index_y = j
                            index_x = k
                            Whether_found = True
                            break
            
            assert (Whether_found == True, "There have other file in loadpath")

            for m in range(0, label.shape[3]):
                tmp = struct.pack('i', arr_read[index_z, index_y, index_x, m])
                outfile.write(tmp)


def Progress_check_label(already_path):
    sub_data_path = os.path.join(already_path)
    if not os.path.exists(sub_data_path):
        os.makedirs(sub_data_path)
    
    already_file_list = sorted(glob.glob(sub_data_path + '/*'))
    already_list = []
    for name_all in already_file_list:
        if len(name_all.split('.')) == 1:
            name = name_all
        elif len(name_all.split('.')) == 2:
            name = name_all.split('.')[0]
            save_suffix = name_all.split('.')[1]
        else:
            break
        already_name = name.split('/')[-1]
        already_list.append(already_name)

    print('Successful resuming of breakpoint')

    return already_list


def get_processing_information(stack_name, stack_path, matrix_path, save_path, raw_path):
    
    
    
    

    Load_PATH = stack_path + stack_name
    
    
    outPutList = []
    outPut = []

    Aff_PATH = matrix_path + stack_name

    OUTPUT_PATH = save_path + stack_name

    
    check_color_path = save_path + stack_name + '/' + 'colorful'

    if not os.path.exists(check_color_path):

        outPut.append(Load_PATH)
        outPut.append(Aff_PATH)
        outPut.append(OUTPUT_PATH)
        outPut.append(raw_path)
        outPutList.append(outPut)

    else:
        stack_names = os.listdir(check_color_path)
        total = int(len(stack_names))  
        if total < 25:
            outPut.append(Load_PATH)
            outPut.append(Aff_PATH)
            outPut.append(OUTPUT_PATH)
            outPut.append(raw_path)
            outPutList.append(outPut)

    return outPutList




def write_big_label(stack_path, save_path, raw_path, overloap_x_pixel, overloap_y_pixel, overloap_z_pixel, start, end):
    a = 1

    load_path = stack_path
    
    save_path = save_path
    raw_path = raw_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    

    
    

    all_stack = os.listdir(load_path)
    all_stack.sort(key=lambda x: int(x[5:]))

    all_stack = all_stack[start:end]
    print(all_stack)

    total_z = int(len(all_stack))  

    num_x_list = []
    num_y_list = []

    for height in range(0, total_z):
        sub_data_path = os.path.join(load_path, all_stack[height])
        sub_tifs = os.listdir(sub_data_path)

        total = int(len(sub_tifs))  

        sub_tifs.sort(
            key=lambda x: int(x.split('-')[0] * 100 + x.split('_')[1] * 10 + x.split('.')[0].split('_')[2] * 1))
        
        sorted_file = sub_tifs


        
        
        
        row_all = np.zeros(total)
        line_all = np.zeros(total)

        
        
        for i, name in enumerate(sorted_file):
            row_all[i] = int(name.split('_')[1])
            
        for i, name in enumerate(sorted_file):
            line_all[i] = int(name.split('.')[0].split('_')[2])
            

        
        number_y = np.unique(row_all)
        number_x = np.unique(line_all)

        
        num_y = len(number_y)
        num_x = len(number_x)

        
        
        



        inital_z, inital_y, inital_x, label = load_binary_dat_break_down(
            save_path + '/' + all_stack[height] + '/' +  all_stack[height] + '/' + "merge2.dat")  
        print('in writing! load ', save_path + '/' + all_stack[height] + '/' + all_stack[height] + '/' + "merge2.dat",       
              ' is successful')

        make_big_label(raw_path + all_stack[height], load_path + all_stack[height], sorted_file,
                       save_path  + '/' + all_stack[height]+ '/' + all_stack[height], label,   
                       num_x, num_y, 1, overloap_x_pixel, overloap_y_pixel, overloap_z_pixel)


def read_h5():
    label_path = '/home/liujz/bigstore3dian33_liujz/temp/banmayu_cellhe/00001-00031_04_05.h5'
    
    with h5py.File(label_path, 'r') as f:
        label = f['data'][:]  
        size_z_y_x = f['shape'][:]

    label = np.int32(label)  
    a = 1



def zbf_cellhe_process_h5(labels):
    
    
    
    
    
    
    
    

    
    
    
    

        

    
    
    
    
    

    import fastremap
    a = np.max(labels)
    b = np.min(labels)

    
    
    
    
    
    
    
    
    
    

    

    whether = labels == 0

    new_label = np.zeros_like(labels, np.uint8)
    new_label[whether] = 1

    new_label = new_label * 255

    new_label = np.uint8(new_label)
    mask_back_2_erode = []

    kernel = np.ones((10,10), dtype=np.uint8)
    for i in range(0, new_label.shape[0]):
        
        mask_back_2_erode_temp = cv2.dilate(new_label[i], kernel)
        mask_back_2_erode.append(mask_back_2_erode_temp)

    mask_back_2_erode = np.stack(mask_back_2_erode, axis=0)

    mask_back_2_erode = np.uint8((255 - mask_back_2_erode)/255)

    labels = labels * mask_back_2_erode
    
    


    connectivity = 26 
    labels = cc3d.connected_components(labels, connectivity=connectivity)

    labels = cc3d.dust(
        labels, threshold=1500,
        connectivity=26, in_place=False
    )


    inverse, packed = np.unique(labels, return_inverse=True)


    a = np.max(labels)
    b = np.min(labels)

    for label_num in inverse:
        if label_num == 0:
            continue
        whether = labels == label_num
        new_label = np.zeros_like(labels, np.uint8)
        new_label[whether] = 1
        new_label = new_label * 255
        new_label = np.uint8(new_label)
        mask_back_2_erode = []
        for i in range(0, new_label.shape[0]):
            
            mask_back_2_erode_temp = cv2.dilate(new_label[i], kernel)
            mask_back_2_erode.append(mask_back_2_erode_temp)

        mask_back_2_erode = np.stack(mask_back_2_erode, axis=0)
        whether = mask_back_2_erode == 255
        labels[whether] = label_num


    
    
    
    
    
    
    
    
    
    


    a=2
    return labels

def zbf_neuro_process_h5(labels):

    import fastremap

    connectivity = 26  
    labels = cc3d.connected_components(labels, connectivity=connectivity)

    labels = cc3d.dust(
        labels, threshold=2000,
        connectivity=26, in_place=False
    )

    a = 2
    return labels

def zbf_cellhe_processed_all(stack_path, save_path, start, end):

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    

    
    

    all_stack = os.listdir(stack_path)
    all_stack.sort(key=lambda x: int(x[5:]))

    all_stack = all_stack[start:end]
    print(all_stack)

    total_z = int(len(all_stack))  

    for height in range(0, total_z):
        sub_data_path = os.path.join(stack_path, all_stack[height])
        sub_tifs = os.listdir(sub_data_path)

        total = int(len(sub_tifs))  

        sub_tifs.sort(
            key=lambda x: int(x.split('-')[0] * 100 + x.split('_')[1] * 10 + x.split('.')[0].split('_')[2] * 1))
        
        sorted_file = sub_tifs

        if not os.path.exists(save_path + '/' + all_stack[height]):
            os.mkdir(save_path + '/' + all_stack[height])

        for hang_line_name in sub_tifs:
            print(save_path + '/' + all_stack[height] + '/' + hang_line_name)
            sub_sub_data_path = save_path + '/' + all_stack[height] + '/' + hang_line_name

            if not os.path.exists(sub_sub_data_path):
                os.mkdir(sub_sub_data_path)

            with h5py.File(stack_path + '/' + all_stack[height] + '/' + hang_line_name + '/' + hang_line_name +'.h5', 'r') as f:
                label = f['data'][:]  
                size_z_y_x = f['shape'][:]

            label = np.int32(label)  

            label = zbf_cellhe_process_h5(label)

            with h5py.File(sub_sub_data_path + '/' + hang_line_name +'.h5', 'w') as f:
                f.create_dataset('data', data=label, compression='gzip')
                f.create_dataset('shape', data=(label.shape[0], label.shape[1], label.shape[2]),
                                 compression='gzip')

            
            



def zbf_neuro_processed_all(stack_path, save_path, start, end):

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    

    
    

    all_stack = os.listdir(stack_path)
    all_stack.sort(key=lambda x: int(x[5:]))

    all_stack = all_stack[start:end]
    print(all_stack)

    total_z = int(len(all_stack))  

    for height in range(0, total_z):
        sub_data_path = os.path.join(stack_path, all_stack[height])
        sub_tifs = os.listdir(sub_data_path)

        total = int(len(sub_tifs))  

        sub_tifs.sort(
            key=lambda x: int(x.split('-')[0] * 100 + x.split('_')[1] * 10 + x.split('.')[0].split('_')[2] * 1))
        
        sorted_file = sub_tifs

        if not os.path.exists(save_path + '/' + all_stack[height]):
            os.mkdir(save_path + '/' + all_stack[height])

        for hang_line_name in sub_tifs:
            print(save_path + '/' + all_stack[height] + '/' + hang_line_name)
            sub_sub_data_path = save_path + '/' + all_stack[height] + '/' + hang_line_name

            if not os.path.exists(sub_sub_data_path):
                os.mkdir(sub_sub_data_path)

            with h5py.File(stack_path + '/' + all_stack[height] + '/' + hang_line_name + '/' + hang_line_name +'.h5', 'r') as f:
                label = f['data'][:]  
                size_z_y_x = f['shape'][:]

            label = np.int32(label)  

            
            label = zbf_neuro_process_h5(label)

            with h5py.File(sub_sub_data_path + '/' + hang_line_name +'.h5', 'w') as f:
                f.create_dataset('data', data=label, compression='gzip')
                f.create_dataset('shape', data=(label.shape[0], label.shape[1], label.shape[2]),
                                 compression='gzip')

            
            


def zbf_neuro_processed_test():
    
    stack_path = '/home/liujz/bigstore3dian33_liujz/temp/for_bei/1/neuro/all/stack169/stack169/00001-00031_06_06/00001-00031_06_06.h5'
    save_path = '/home/liujz/bigstore3dian33_liujz/temp/for_bei/1/neuro/all/stack169/stack169/00001-00031_06_06_norm'

    if not os.path.exists(save_path):
        os.mkdir(save_path)


    with h5py.File(stack_path, 'r') as f:
        label = f['data'][:]  
        size_z_y_x = f['shape'][:]

    label = np.int32(label)  

    
    

    a = np.max(label)

    
    
    
    

    for i in range(0, label.shape[0]):
        imageio.imwrite(save_path + '/' + str(i + 1).zfill(4) + '.tif', label[i])

def check_same_label():
    label_path = '/home/liujz/bigstore3dian33_liujz/temp/for_bei/1/he/all_singal_merge2/stack173/stack173/big_label3'
    label = read_32_tif_seqqence(label_path)
    

    label_213 = np.where(label == 213, 1, 0)
    label_213 = np.uint8(label_213)
    label_213_max = np.max(label_213)

    label_225 = np.where(label == 225, 2, 0)
    label_225 = np.uint8(label_225)
    label_225_max = np.max(label_225)

    label_232 = np.where(label == 232, 3, 0)
    label_232 = np.uint8(label_232)
    label_232_max = np.max(label_232)

    if label_225_max != 0 :
        whether_225 =  label_225 == 2
        label_213[whether_225] = 2

    if label_232_max != 0 :
        whether_232 =  label_232 == 3
        label_213[whether_232] = 3

    save_path = '/home/liujz/bigstore3dian33_liujz/temp/for_bei/1/he/all_singal_merge2/stack173/stack173/big_label_213'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(0, label_213.shape[0]):
       imageio.imwrite(save_path + '/' + str(i + 1).zfill(4) + '.tif', label_213[i])

    a = 1


def turn_h5_tif():
    stack_path = '/home/liujz/bigstore3dian33_liujz/temp/for_bei/1/he/all2/stack172/'
    sub_tifs = os.listdir(stack_path)
    for label_name in sub_tifs:
        with h5py.File(stack_path + '/' + label_name + '/' + label_name + '.h5', 'r') as f:
            label = f['data'][:]  
            size_z_y_x = f['shape'][:]

        label = np.int32(label)  

        save_path = stack_path + label_name + '/' + label_name
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i in range(0, label.shape[0]):
            imageio.imwrite(save_path + '/' + str(i + 1).zfill(4) + '.tif', label[i])


def copy_merge_dat():
    import shutil

    origin_path = '/home/liujz/liujz/2022_zbf_thire/zbf_cellhe_merge2/'
    save_path = '/home/liujz/liujz/2022_zbf_thire/zbf_cellhe_merge6/'
    stack_names = os.listdir(origin_path)
    stack_names.sort(key=lambda x: int(x[5:]))
    for stack_name in stack_names:
        merge_path = origin_path + '/' + stack_name + '/' + stack_name + '/' + 'merge.dat'
        if not os.path.exists(save_path + '/' + stack_name):
            os.mkdir(save_path + '/' + stack_name)
        if not os.path.exists(save_path + '/' + stack_name + '/' + stack_name):
            os.mkdir(save_path + '/' + stack_name + '/' + stack_name)
        target_path = save_path + '/' + stack_name + '/' + stack_name + '/' + 'merge.dat'
        shutil.copy(merge_path, target_path)


if __name__ == '__main__':
    
    
    
    stack_path = '/opt/data/Nas402/zbrfish/zbfStackResults_all_multicut_cc3d/'
    
    
    
    save_path = '/opt/data/Nas402/zbrfish/zbf_merge_neuro/'
    
    
    raw_path = '/opt/data/Nas402/zbrfish/zbf_all_stackresults_enhance/'
    
    text_path = '/opt/data/Nas402/zbrfish/zbf-txt-idx/zbf-txt-idx/zbf_all_stackresults/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

        
        
        

    overloap_x_pixel = 600 
    overloap_y_pixel = 600 
    overloap_z_pixel = 1
    bite = 19  
    cuda_num = 0
    
    width = 50 


    
    whether_update = False
    whether_save_breakpoint = True

    start = 0
    end = 50
    
    
    merge_tif_bigdata_not_Regular_stack_merge_z(stack_path, save_path, raw_path, text_path, overloap_x_pixel, overloap_y_pixel, overloap_z_pixel, bite, cuda_num, width, start, end)    
    
    #inital_z, inital_y, inital_x, label1 = load_binary_dat_break_down(save_path + '/' + 'stack2' + '/' + 'stack2' + '/' + "merge.dat")
    #inital_z, inital_y, inital_x, label1 = load_binary_dat_break_down(save_path + '/' + 'stack1' + '/' + 'stack1' + '/' + "merge.dat")  
    #print('load ', save_path + '/' + 'stack1' + '/' + 'stack1' + '/' + "merge.dat", ' is successful')
    
    
    #for i in range(1,764):
    #    num = str(i)
    #    name = 'stack' + num
    #    print(name)
    #    inital_z, inital_y, inital_x, label1 = load_binary_dat_break_down(save_path + '/' + name + '/' + name + '/' + "merge.dat")  
    #    #print('load ', save_path + '/' + 'stack1' + '/' + 'stack1' + '/' + "merge.dat", ' is successful')
    
    
    #try:
    #    if not os.path.exists(save_path + '/' + 'stack2' + '/' + 'stack2' + '/' + "merge2.dat"):
    #        inital_z, inital_y, inital_x, label1 = load_binary_dat_break_down(save_path + '/' + 'stack2' + '/' + 'stack2' + '/' + "merge.dat")    
    #        print('load ' , save_path + '/' + 'stack2' + '/' + 'stack2' + '/' + "merge.dat" , ' is successful')
    #    else:
    #        inital_z, inital_y, inital_x, label1 = load_binary_dat_break_down(save_path + '/' + 'stack2' + '/' + 'stack2' + '/' + "merge2.dat")  
    #        print('load ', save_path + '/' + 'stack2' + '/' + 'stack2' + '/' + "merge2.dat", ' is successful')
    #except:
    #    print(save_path + '/' + 'stack2' + '/' + 'stack2' + " merge.dat is not existing!")
        

    
    

    
    

    

    

    

    
    

    
    



    

    


