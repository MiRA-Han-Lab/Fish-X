import numpy as np


import tifffile
from libtiff import TIFF
import imageio
import math
from pairwise_match_ljz_label_cuda_new import pair_match
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

def overlap_fusing(img1, img2, direction, halo_size, whether_label):

    if whether_label == False:

        if direction == 3:
            for num in range(0, img2.shape[0]):

                for j in range(halo_size):
                    alpha = (halo_size - j) / halo_size
                    img2[num, :, j] = alpha * img1[num, :, img1.shape[2] - halo_size + j] + (1 - alpha) * img2[num, :, j]

        if direction == 2:
            for num in range(0, img2.shape[0]):

                for j in range(halo_size):
                    alpha = (halo_size - j) / halo_size
                    img2[num, j, :] = alpha * img1[num, img1.shape[1] - halo_size + j, :] + (1 - alpha) * img2[num, j, :]

        if direction == 1:
            for j in range(halo_size):
                alpha = (halo_size - j) / halo_size
                img2[j, :, :] = alpha * img1[img1.shape[0] - halo_size + j,:, :] + (1 - alpha) * img2[j, :, :]

    else:
        pass

    return img1, img2

def make_label_s(block1,block2,label, height, hang, line, direction):

    
    
    
    
    
    

    

    
    
    
    

    
    
    
    


    
    
    

    uq, s, p = np.unique(block2, return_index=True, return_inverse=True)
    label1_max = np.max(block1)
    index = p.reshape(block2.shape)
    label_new = np.arange(start=label1_max + 1, stop=label1_max + 1 + uq.shape[0], step=1, dtype='int32')
    label_new[uq == 0] = 0
    block2_new = label_new[index]

    s0=  s // (block2.shape[1] * block2.shape[2])
    s0_yu=s % (block2.shape[1] * block2.shape[2])
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
        for num_line in range(0,line):
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

def make_label(block1, block2, max_block1_part_all, max_block2_part_all, label, height, hang, line,  direction):
    
    
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

def make_label_new(block1,block2):

    label1_max = np.max(block1)

    block2_new = [x + label1_max for x in block2]
    
    block2_new = np.where(block2_new == label1_max, 0, block2_new)

    
    
    
    
    



    return block1, block2_new

def unique_label(label1_max,block2):

    label2, index = np.unique(block2, return_inverse=True)
    index = index.reshape(block2.shape)
    label2_new = np.arange(start=label1_max+1, stop=label1_max+1+label2.shape[0], step=1, dtype='uint64')
    label2_new[label2 == 0] = 0
    block2_new = label2_new[index]

    return block2_new


def mul_tif_read(tif_loc):
    tif = TIFF.open(tif_loc, mode='r')
    imgs = np.array(list(tif.iter_images()))

    
    

    assert imgs.dtype == np.uint8, 'Need to be uint8 of numpy array.'
    assert np.max(np.logical_and(imgs < 255, imgs > 0)) == False, 'Need to be segmentation with only 0 and 255.'

    return imgs

def get_one_block_size(load_path, whether_loadpath = True):
    
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

    if  len(data_path.split('.')) == 1:

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

def get_one_block_size_default(load_path, whether_loadpath = True):
    
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
    if len(images)>10:
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

    print('example: ',  address + '/' + 'layer' + str(order_start) + '_' + str(order_y) + '_' + str(order_x) + '.' + suffix)

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




def read_32_tif_seqqence_part(image_dir, whether_line, whether_row, whether_height, halo_size, front, label, height, hang, line):
    

    print('image_dir: ' + image_dir)

    h5_files = os.listdir(image_dir)
    image_dir = os.path.join(image_dir, h5_files[0])
    
    
    test_data_path = os.path.join(image_dir)

    if len(image_dir.split('.'))==1:

        images = os.listdir(test_data_path)

        if len(images)>10:

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

                            image_temp = np.array(pil_image.crop([kuan-halo_size,0,kuan,chang]))     
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

                            image_temp = np.array(pil_image.crop([0,0,halo_size,chang]))     
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

                            image_temp = np.array(pil_image.crop([0, chang-halo_size, kuan, chang]))  
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

                        if i == halo_size-1:
                            break
        else:
            file_name = []
            if  'seg_inv.h5' in images:
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

                    img = labels[:,:,kuan - halo_size:]

                else:

                    img = labels[:, :, :halo_size]

            if whether_row == True:
                if front == True:

                    img = labels[:,chang - halo_size:,:]

                else:

                    img = labels[:, :halo_size:, :]

            if whether_height == True:
                if front == True:

                    img = labels[num - halo_size:,:,:]

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
def label_update(block,label, height, hang, line):

    Update=False
    for num, value in enumerate(label[height][hang][line]):
        if num!=value:
            Update=True
            break
    if Update==True:
        for i in range(0,block.shape[0]):
            for j in range(0, block.shape[1]):
                for k in range(0, block.shape[2]):
                    block[i,j,k]=label[height][hang][line][block[i,j,k]]
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

def img_upadte_all(FilenameAray, load_path,sorted_file ,save_path, label, num_x, num_y, num_z):


    print('begin write')
    for file in sorted_file:

        
        
        

        image_dir=load_path + '/' + file
        save=save_path + '/' + file

        if not os.path.exists(save):
            os.mkdir(save)


        for i in range(0, num_z):
            for j in range(0, num_y):
                for k in range(0, num_x):
                    if FilenameAray[i][j][k] == file:
                        img_upadte(image_dir, save, label, i, j, k, whether_h5=True)
        
        
        
        
        



def img_update_from_binary(load_path, save_path ,dat_path):

    
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


def save_binary_dat(label, save_path, bite, num_z, num_y, num_x, overloap_z_pixel, overloap_y_pixel, overloap_x_pixel, size_z, size_y, size_x, height, hang, line):

    
    with open(save_path + '/' + "merge.dat", "wb") as outfile:  
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

        arr_read = struct.unpack('{}i'.format((int(num_x[0])*int(num_y[0])*int(num_z[0])) * (2 ** int(bite[0]))), f.read())  
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

        arr_read = struct.unpack('{}i'.format((int(num_x[0]) * int(num_y[0]) * int(num_z[0])) * (2 ** int(bite[0]))), f.read())  
        arr_read = np.array(list(arr_read), dtype=np.int32)
        arr_read = arr_read.reshape(label.shape)

    return int(height[0]), int(hang[0]), int(line[0]), arr_read



def read_chongdie_area_all(FilenameAray, SizeAray, load_path, sorted_file, num_x, num_y, num_z, whether_line,
                           whether_row, whether_height, height, hang, line, label, overloap_x_pixel, overloap_y_pixel,
                           halo_size):
    

    if whether_line == True:

        max_block1_part_all = 0
        max_block2_part_all = 0

        if FilenameAray[height][hang][line - 1] == '0':
            block1_part = np.zeros(
                (SizeAray[height][hang][line - 1][0], SizeAray[height][hang][line - 1][1], halo_size), dtype=np.int32)
        else:
            blcok_name = FilenameAray[height][hang][line - 1]
            line_first_name = load_path + '/' + blcok_name
            block1_part, max_block1_part = read_32_tif_seqqence_part(line_first_name, True, False, False, halo_size, True, label, height, hang, line - 1)
            

            if max_block1_part > max_block1_part_all:
                max_block1_part_all = max_block1_part

        if FilenameAray[height][hang][line] == '0':
            block2_part = np.zeros((SizeAray[height][hang][line][0], SizeAray[height][hang][line][1], halo_size),
                                   dtype=np.int32)
        else:
            blcok_name = FilenameAray[height][hang][line]
            line_next_name = load_path + '/' + blcok_name
            block2_part, max_block2_part = read_32_tif_seqqence_part(line_next_name, True, False, False, halo_size,
                                                                     False, label, height, hang, line)
            

            if max_block2_part > max_block2_part_all:
                max_block2_part_all = max_block2_part

    if whether_row == True:

        max_block1_part_all = 0
        max_block2_part_all = 0

        
        if FilenameAray[height][hang - 1][0] == '0':
            block1_part = np.zeros((SizeAray[height][hang - 1][0][0], halo_size, SizeAray[height][hang - 1][0][2]),
                                   dtype=np.int32)
            max_block1_part = 0
        else:
            row_first_name = load_path + '/' + FilenameAray[height][hang - 1][0]
            block1_part, max_block1_part = read_32_tif_seqqence_part(row_first_name, False, True, False, halo_size,
                                                                     True, label, height, hang - 1, 0)
            

            if max_block1_part > max_block1_part_all:
                max_block1_part_all = max_block1_part

        for i in range(1, num_x):
            if FilenameAray[height][hang - 1][i] == '0':
                second_part = np.zeros((SizeAray[height][hang - 1][i][0], halo_size, SizeAray[height][hang - 1][i][2]),
                                       dtype=np.int32)
                
                block1_part = np.concatenate([block1_part, second_part[:, :, overloap_x_pixel:]], axis=2)
                max_block1_part = 0

                
                

            else:
                second_name = load_path + '/' + FilenameAray[height][hang - 1][i]
                second_part, max_block1_part = read_32_tif_seqqence_part(second_name, False, True, False, halo_size,
                                                                         True, label, height, hang - 1, i)
                

                
                block1_part = np.concatenate([block1_part[:, :, :-overloap_x_pixel], second_part], axis=2)

                if max_block1_part > max_block1_part_all:
                    max_block1_part_all = max_block1_part

        if FilenameAray[height][hang][0] == '0':
            block2_part = np.zeros((SizeAray[height][hang][0][0], halo_size, SizeAray[height][hang][0][2]),
                                   dtype=np.int32)

            max_block2_part = 0

        else:
            row_next_name = load_path + '/' + FilenameAray[height][hang][0]
            block2_part, max_block2_part = read_32_tif_seqqence_part(row_next_name, False, True, False, halo_size,
                                                                     False, label, height, hang, 0)
            

            if max_block2_part > max_block2_part_all:
                max_block2_part_all = max_block2_part

        for i in range(1, num_x):

            if FilenameAray[height][hang][i] == '0':
                second_part = np.zeros((SizeAray[height][hang][i][0], halo_size, SizeAray[height][hang][i][2]),
                                       dtype=np.int32)
                
                block2_part = np.concatenate([block2_part, second_part[:, :, overloap_x_pixel:]], axis=2)
                max_block2_part = 0

            else:
                second_name = load_path + '/' + FilenameAray[height][hang][i]
                second_part, max_block2_part = read_32_tif_seqqence_part(second_name, False, True, False, halo_size,
                                                                         False, label, height, hang, i)
                

                
                block2_part = np.concatenate([block2_part[:, :, :-overloap_x_pixel], second_part], axis=2)

                if max_block2_part > max_block2_part_all:
                    max_block2_part_all = max_block2_part

    if whether_height == True:

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
                                                                        True, label, height - 1, hang, 0)
                

                if max_block1_part > max_block1_part_all:
                    max_block1_part_all = max_block1_part

            for line in range(1, num_x):

                if FilenameAray[height - 1][hang][line] == '0':
                    hang_next = np.zeros(
                        (halo_size, SizeAray[height - 1][hang][line][1], SizeAray[height - 1][hang][line][2]),
                        dtype=np.int32)
                    hang_first = np.concatenate([hang_first, hang_next], axis=2)

                    max_block1_part = 0

                else:
                    hang_next_name = load_path + '/' + FilenameAray[height - 1][hang][line]
                    hang_next, max_block1_part = read_32_tif_seqqence_part(hang_next_name, False, False, True,
                                                                           halo_size, True, label, height - 1, hang,
                                                                           line)
                    

                    hang_first = np.concatenate([hang_first, hang_next], axis=2)

                    if max_block1_part > max_block1_part_all:
                        max_block1_part_all = max_block1_part

            if hang == 0:
                block1_part = hang_first
            else:
                block1_part = np.concatenate([block1_part, hang_first], axis=1)

        del hang_first

        

        for hang in range(0, num_y):

            if FilenameAray[height][hang][0] == '0':
                hang_first = np.zeros((halo_size, SizeAray[height][hang][0][1], SizeAray[height][hang][0][2]),
                                      dtype=np.int32)
                max_block2_part = 0

            else:
                hang_first_name = load_path + '/' + FilenameAray[height][hang][0]
                hang_first, max_block2_part = read_32_tif_seqqence_part(hang_first_name, False, False, True, halo_size,
                                                                        False, label, height, hang, 0)
                

                if max_block2_part > max_block2_part_all:
                    max_block2_part_all = max_block2_part

            for line in range(1, num_x):

                if FilenameAray[height][hang][line] == '0':
                    hang_next = np.zeros((halo_size, SizeAray[height][hang][line][1], SizeAray[height][hang][line][2]),
                                         dtype=np.int32)
                    hang_first = np.concatenate([hang_first, hang_next], axis=2)
                else:
                    hang_next_name = load_path + '/' + FilenameAray[height][hang][line]
                    hang_next, max_block2_part = read_32_tif_seqqence_part(hang_next_name, False, False, True,
                                                                           halo_size, False, label, height, hang, line)
                    

                    hang_first = np.concatenate([hang_first, hang_next], axis=2)

                    if max_block2_part > max_block2_part_all:
                        max_block2_part_all = max_block2_part

            if hang == 0:
                block2_part = hang_first
            else:
                block2_part = np.concatenate([block2_part, hang_first], axis=1)

        del hang_first

    return block1_part, block2_part, max_block1_part_all, max_block2_part_all

def compute_matrix(matrix_cell, i):
    if i == 1:
        matrix = matrix_cell[i - 1]
    else:
        matrix = np.linalg.multi_dot(matrix_cell[0:i])
    a_pinv = np.linalg.pinv(matrix)
    a_pinv[2,0]=0
    a_pinv[2,1]=0
    a_pinv[2,2]=1
    
    return a_pinv


def Affine_transformation(FilenameAray, SizeAray, block1_part, block2_part, affine_address, affine_scale, height, hang, line, whether_line, whether_row, whether_height, pad_size, order):

    

    if whether_line == True:

        block1_name = FilenameAray[height][hang][line-1].split('.')[0]
        block2_name = FilenameAray[height][hang][line].split('.')[0]

        if block1_name == '0':
            block1_out = np.pad(block1_part, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')
        else:
            pad_block1_part = np.pad(block1_part, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')
            
            
            
            affine_blcok1_matrix = affine_address + "/" + '/affine_' + block1_name.split('_')[1].zfill(2) + block1_name.split('.')[0].split('_')[2].zfill(2) + '.mat'
            
            matrix_cell_block1 = sio.loadmat(affine_blcok1_matrix)['affine'][0]
            matrix_cell_block1 = matrix_cell_block1[-pad_block1_part.shape[0] + 1:]
            for i in range(matrix_cell_block1.shape[0] - 1):
                
                matrix_cell_block1[i][0, 2] = matrix_cell_block1[i][0, 2] * affine_scale 
                matrix_cell_block1[i][1, 2] = matrix_cell_block1[i][1, 2] * affine_scale 

                
                error_index = matrix_cell_block1[i][0, 0] * matrix_cell_block1[i][1, 1] - matrix_cell_block1[i][0, 1] * \
                              matrix_cell_block1[i][1, 0]
                if error_index > 1.1 or error_index < 0.9:
                    print('error, change %d' % i)
                    matrix_cell_block1[i] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                


        if block2_name == '0':
            block2_out = np.pad(block2_part, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')
        else:
            pad_block2_part = np.pad(block2_part,((0,0),(pad_size,pad_size),(pad_size,pad_size)),'constant')

            if block1_name == '0':
                
                affine_blcok2_matrix = affine_address + "/" + '/affine_' + block2_name.split('_')[1].zfill(2) + block2_name.split('.')[0].split('_')[2].zfill(2) + '.mat'
            else:
                
                affine_blcok2_matrix = affine_address + "/" + '/affine_' + block1_name.split('_')[1].zfill(2) + block1_name.split('.')[0].split('_')[2].zfill(2) + '.mat'

            

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
                

        matrix_cell_block=[]
        if block2_name == '0' and block1_name == '0':
            del block1_part, block2_part

            return block1_out, block2_out
        else:
            if block2_name == '0':
                matrix_cell_block = matrix_cell_block1
            if block1_name == '0':
                matrix_cell_block = matrix_cell_block2
            if block2_name != '0' and block1_name != '0':
                matrix_cell_block = (matrix_cell_block1 + matrix_cell_block2)/2

            if block1_name!='0':
                
                block1_out = np.zeros_like(pad_block1_part)
                for i in range(pad_block1_part.shape[0]):
                    im = pad_block1_part[i].T
                    if i == 0:
                        block1_out[i] = im.T
                    else:
                        matrix = compute_matrix(matrix_cell_block, i)
                        block1_out[i] = affine_transform(im, matrix, order=order).T
                
                
                
                
                
                print('finish '+ block1_name + 'line_affine_transform!')

            if block2_name!='0':
                
                block2_out = np.zeros_like(pad_block2_part)
                for i in range(pad_block2_part.shape[0]):
                    im = pad_block2_part[i].T
                    if i == 0:
                        block2_out[i] = im.T
                    else:
                        matrix = compute_matrix(matrix_cell_block, i)   
                        block2_out[i] = affine_transform(im, matrix, order=order).T

                
                
                
                
                
                print('finish ' + block2_name + 'line_affine_transform!')

            del block1_part,block2_part

            return block1_out, block2_out

    elif whether_row == True:

        block1_name_all = FilenameAray[height][hang-1]
        block2_name_all = FilenameAray[height][hang]

        pad_block1_part = np.pad(block1_part, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')
        pad_block2_part = np.pad(block2_part, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')
        del block1_part, block2_part

        matrix_cell_block1_allfinal = np.zeros((len(FilenameAray[height][hang-1]), SizeAray[height][hang-1][0][0]-1, 3, 3),dtype=np.float)
        block1_whether_zero = np.zeros((len(FilenameAray[height][hang-1])),dtype=np.float)
        matrix_cell_block2_allfinal = np.zeros((len(FilenameAray[height][hang]), SizeAray[height][hang][0][0]-1, 3, 3),dtype=np.float)
        block2_whether_zero = np.zeros((len(FilenameAray[height][hang])), dtype=np.float)

        for i in range(len(FilenameAray[height][hang-1])):

            block1_name = block1_name_all[i].split('.')[0]
            block2_name = block2_name_all[i].split('.')[0]

            if block1_name == '0':
                matrix_cell_block1 = np.zeros((SizeAray[height][hang-1][0][0]-1, 3, 3),dtype=np.float)
                matrix_cell_block1_allfinal[i] = matrix_cell_block1
                block1_whether_zero[i] = 0
            else:
                
                
                affine_blcok1_matrix = affine_address + "/" + '/affine_' + block1_name.split('_')[1].zfill(2) + block1_name.split('.')[0].split('_')[2].zfill(2) + '.mat'

                matrix_cell_block1 = sio.loadmat(affine_blcok1_matrix)['affine'][0]
                matrix_cell_block1 = matrix_cell_block1[-pad_block1_part.shape[0] + 1:]
                for j in range(matrix_cell_block1.shape[0] - 1):
                    matrix_cell_block1_allfinal[i][j] = matrix_cell_block1[j]
                block1_whether_zero[i] = 1

            if block2_name == '0':
                matrix_cell_block2 = np.zeros((SizeAray[height][hang][0][0]-1, 3, 3),dtype=np.float)
                matrix_cell_block2_allfinal[i] = matrix_cell_block2
                block2_whether_zero[i] = 0
            else:
                if block1_name == '0':
                    
                    affine_blcok2_matrix = affine_address + "/" + '/affine_' + block2_name.split('_')[1].zfill(2) + block2_name.split('.')[0].split('_')[2].zfill(2) + '.mat'
                else:
                    
                    affine_blcok2_matrix = affine_address + "/" + '/affine_' + block1_name.split('_')[1].zfill(2) + block1_name.split('.')[0].split('_')[2].zfill(2) + '.mat'
                
                matrix_cell_block2 = sio.loadmat(affine_blcok2_matrix)['affine'][0]
                matrix_cell_block2 = matrix_cell_block2[-pad_block2_part.shape[0] + 1:]
                for j in range(matrix_cell_block2.shape[0] - 1):
                    matrix_cell_block2_allfinal[i][j] = matrix_cell_block2[j]
                block2_whether_zero[i] = 1

        
        matrix_cell_block1 = np.zeros((3, 3),dtype=np.float)
        matrix_cell_block2 = np.zeros((3, 3),dtype=np.float)
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
            

        matrix_cell_block = (matrix_cell_block1 + matrix_cell_block2)/2

        
        
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

        block1_name = FilenameAray[height-1]
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


def merge_tif_bigdata_not_Regular_stack(x, overloap_x_pixel, overloap_y_pixel, overloap_z_pixel, bite, cuda_num = 0, width = 15, affine_scale = 10, whether_update = False, whether_save_breakpoint = False):

    load_path = x[0]
    aff_path = x[1]
    save_path = x[2]
    raw_path = x[3]

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    sub_data_path = os.path.join(load_path)
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

    num_z = len(number_z)
    num_y = len(number_y)
    num_x = len(number_x)

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
        name_line = int(name.split('.')[0].split('_')[2])
        name_height = int(name.split('-')[0])
        
        
        

        z = all_z.index(name_height)
        y = all_y.index(name_row)
        x = all_x.index(name_line)

        
        
        FilenameAray[z][y][x] = name


    
    size_z, size_y, size_x = get_one_block_size(load_path, whether_loadpath = True)
    print('block size_z:', size_z)
    print('block size_y:', size_y)
    print('block size_x:', size_x)
    

    
    #if (os.path.exists(save_path + '/' + "merge.dat") and whether_save_breakpoint==True):
    if (os.path.exists(save_path + '/' + "merge.dat")):
        inital_z, inital_y, inital_x, label = load_binary_dat_break_down(save_path + '/' + "merge.dat")         
    else:
        inital_z = 0
        inital_y = 0
        inital_x = 1
        
        label = np.zeros((num_z, num_y, num_x, 2 ** bite), dtype=np.int32)
        for i in range(0, num_z):
            for j in range(0, num_y):
                for k in range(0, num_x):
                    label[i, j, k, :] = np.arange(2 ** bite).astype(np.int32)
    

    
    affine_address = aff_path

    

    SizeAray = []
    SizeAray_xy = []
    for j in range(num_z):
        SizeAray_xy = []
        for i in range(num_y):
            zeroArray = [[0,0,0] for i in range(num_x)]
            SizeAray_xy.append(zeroArray)
        SizeAray.append(SizeAray_xy)

    
    for height in range(0, num_z):
        for hang in range(0, num_y):
            for line in range(0, num_x):
                if FilenameAray[height][hang][line] == '0':
                    pass
                else:
                    
                    size_z, size_y, size_x = get_one_block_size(load_path + '/' + FilenameAray[height][hang][line], whether_loadpath=False)
                    
                    
                    
                    
                    
                    SizeAray[height][hang][line][0] = size_z
                    SizeAray[height][hang][line][1] = size_y
                    SizeAray[height][hang][line][2] = size_x
    
    for height in range(0, num_z):
        for hang in range(0, num_y):
            for line in range(0, num_x):
                if FilenameAray[height][hang][line] != '0':
                    pass
                else:
                    for i in range(0, num_x):
                        if FilenameAray[height][hang][i] != '0':
                            SizeAray[height][hang][line][1] = SizeAray[height][hang][i][1]
                            SizeAray[height][hang][line][0] = SizeAray[height][hang][i][0]
                            break
                    for j in range(0, num_y):
                        if FilenameAray[height][j][line] != '0':
                            SizeAray[height][hang][line][2] = SizeAray[height][j][line][2]
                            break
    

    
    print(FilenameAray)
    print(SizeAray)

    height = 0
    hang = 0
    line = 1

    max_height_last = 0
    for height in range(0, num_z):
        max_hang_last = 0
        for hang in range(0, num_y): 
            max_line_last = 0
            for line in range(1, num_x):

                if (height * 100 + hang * 10 + line * 1) < (inital_z * 100 + inital_y * 10 + inital_x * 1):
                    continue

                block1_part, block2_part, max_block1_part_all, max_block2_part_all = read_chongdie_area_all(FilenameAray, SizeAray, load_path, sorted_file, num_x, num_y, num_z, True, False, False, height, hang, line, label, overloap_x_pixel, overloap_y_pixel, halo_size = overloap_x_pixel)

                
                if max_line_last <= max_block1_part_all:
                    max_line_last = max_block1_part_all
                elif max_block1_part_all == 0:
                    print('block1_part x ', FilenameAray[height][hang][line - 1], ' is a blank block')
                else:
                    print('max_block1_part_all: ', max_block1_part_all)
                    print('max_line_last: ', max_line_last)
                    print('block1_part x ', FilenameAray[height][hang][line - 1], ' get max_block1_part_all < max_line_last')
                    
                

                block1_part, block2_part = Affine_transformation(FilenameAray, SizeAray, block1_part, block2_part, affine_address, affine_scale, height, hang, line, True, False, False, 100, order = 0)

                row_new, row_next, label = make_label(block1_part, block2_part, max_line_last, max_block2_part_all, label, height, hang, line, direction=3) 
                
                label = pair_match(row_new, row_next,  height, hang, line, label, direction=3, halo_size = overloap_x_pixel, cuda_num = cuda_num, width = width)

                if whether_save_breakpoint == True:
                    height_next, hang_next, line_next = get_next_for_per(height, hang, line, 0, 0, 1)
                    save_binary_dat(label, save_path, bite, num_z, num_y, num_x, overloap_z_pixel, overloap_y_pixel, overloap_x_pixel, size_z, size_y, size_x, height_next, hang_next, line_next)

            if hang == 0:
                continue
            if (height * 100 + hang * 10 + line * 1) < (inital_z * 100 + inital_y * 10 + (inital_x - 1) * 1):
                continue

            block1_part, block2_part, max_block1_part_all, max_block2_part_all = read_chongdie_area_all(FilenameAray, SizeAray, load_path, sorted_file, num_x, num_y, num_z, False, True, False, height, hang, 0, label, overloap_x_pixel, overloap_y_pixel, halo_size = overloap_y_pixel)

            
            if max_hang_last <= max_block1_part_all:
                max_hang_last = max_block1_part_all
            elif max_block1_part_all == 0:
                print('block1_part y is a blank block')
            else:
                print('block1_part y get max_block1_part_all error')
                
            


            
            

            block1_part, block2_part = Affine_transformation(FilenameAray, SizeAray, block1_part, block2_part, affine_address, affine_scale, height, hang, line, False, True, False, 200, order = 0)

            
            

            row_new, row_next, label = make_label(block1_part, block2_part, max_hang_last, max_block2_part_all,  
                                                  label, height, hang, line=num_x,
                                                  direction=2)
            
            label = pair_match(row_new, row_next, height, hang, num_x, label, direction=2, halo_size = overloap_y_pixel, cuda_num = cuda_num, width = width)

            if whether_save_breakpoint == True:
                height_next, hang_next, line_next = get_next_for_per(height, hang, line, 0, 1, 0)
                save_binary_dat(label, save_path, bite, num_z, num_y, num_x, overloap_z_pixel, overloap_y_pixel, overloap_x_pixel, size_z, size_y, size_x, height_next, hang_next, line_next)

        if height == 0:
            continue
        if (height * 100 + hang * 10 + line * 1) < (inital_z * 100 + (inital_y - 1) * 10 + inital_x * 1):
            continue

        block1_part, block2_part, max_block1_part_all, max_block2_part_all = read_chongdie_area_all(FilenameAray, SizeAray, load_path, sorted_file, num_x, num_y, num_z, False, False, True, height, 0,  0, label, overloap_x_pixel, overloap_y_pixel, halo_size=overloap_z_pixel)

        
        if max_height_last <= max_block1_part_all:
            max_height_last = max_block1_part_all
        elif max_block1_part_all == 0:
            print('block1_part z is a blank block')
        else:
            print('block1_part z get max_block1_part_all error')
            
        


        row_new, row_next, label = make_label(block1_part, block2_part, max_height_last, max_block2_part_all, label,   
                                              height, hang=num_y, line=num_x,
                                              direction=1)
        
        label = pair_match(row_new, row_next, height, num_y, num_x, label, direction=1, halo_size=overloap_z_pixel, cuda_num = cuda_num, width = width)

        if whether_save_breakpoint == True:
            height_next, hang_next, line_next = get_next_for_per(height, hang, line, 1, 0, 0)
            save_binary_dat(label, save_path, bite, num_z, num_y, num_x, overloap_z_pixel, overloap_y_pixel, overloap_x_pixel, size_z, size_y, size_x, height_next, hang_next, line_next)


    if whether_update == True:
        save_binary_dat(label, save_path, bite, num_z, num_y, num_x, overloap_z_pixel, overloap_y_pixel, overloap_x_pixel, size_z, size_y, size_x, height + 1, hang, line)
        img_upadte_all(FilenameAray, load_path, sorted_file, save_path, label, num_x, num_y, num_z)
    else:
        save_binary_dat(label, save_path, bite, num_z, num_y, num_x, overloap_z_pixel, overloap_y_pixel, overloap_x_pixel, size_z, size_y, size_x, height + 1, hang, line)


    


def make_big_label(raw_path, load_path, sorted_file, save_path, label, num_x, num_y, num_z, overloap_x_pixel, overloap_y_pixel, overloap_z_pixel):
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
            row_first_name_raw = raw_path + '/' + str(sorted_file[hang * num_x + height * num_x * num_y])
            
            
            row_new_raw = read_chen_z_y_x(row_first_name_raw)
            row_new_raw = np.uint8(row_new_raw)
            
            print('raw read: ' + row_first_name_raw)
            print('row_new_raw shape:', row_new_raw.shape)

            
            row_new_label = read_32_tif_seqqence(row_first_name_label)
            row_new_label = label_update(row_new_label, label, height, hang, 0)
            print('label update: ' + row_first_name_label)
            print('label update shape:', row_new_label.shape)


            row_new_raw = zoom(row_new_raw, [1, 0.125, 0.125], order=0)
            


            for line in range(1, num_x):
                print('line: ' + str(line))
                row_next_name_label = load_path + '/' + str(sorted_file[hang * num_x + line + height * num_x * num_y])
                row_next_name_raw = raw_path + '/' + str(sorted_file[hang * num_x + line + height * num_x * num_y])
                
                
                
                

                
                row_next_raw = read_chen_z_y_x(row_next_name_raw)
                row_next_raw = np.uint8(row_next_raw)
                print('raw read: ' + row_next_name_raw)
                print('row_new_raw shape:', row_next_raw.shape)
                
                
                row_next_label = read_32_tif_seqqence(row_next_name_label)
                row_next_label = label_update(row_next_label, label, height, hang, line)
                print('label update: ' + row_next_name_label)
                print('label update shape:', row_next_label.shape)


                row_next_raw = zoom(row_next_raw, [1, 0.125, 0.125], order=0)
                


                
                row_new_raw, row_next_raw = overlap_fusing(row_new_raw, row_next_raw, direction=3, halo_size=overloap_x_pixel,
                                                   whether_label = False)

                row_new_raw = np.concatenate([row_new_raw, row_next_raw[:, :, overloap_x_pixel:]], axis=2)

                
                row_new_label, row_next_label = overlap_fusing(row_new_label, row_next_label, direction=3, halo_size=overloap_x_pixel,
                                                   whether_label = True)

                row_new_label = np.concatenate([row_new_label, row_next_label[:, :, overloap_x_pixel:]], axis=2)



            if hang == 0:
                ROWs_raw = row_new_raw
                ROWs_label = row_new_label
            else:
                
                ROWs_raw, row_new_raw = overlap_fusing(ROWs_raw, row_new_raw, direction=2, halo_size=overloap_y_pixel,
                                               whether_label = False)

                ROWs_raw = np.concatenate([ROWs_raw, row_new_raw[:, overloap_y_pixel:, :]], axis=1)

                
                ROWs_label, row_new_label = overlap_fusing(ROWs_label, row_new_label, direction=2, halo_size=overloap_y_pixel,
                                               whether_label = True)

                ROWs_label = np.concatenate([ROWs_label, row_new_label[:, overloap_y_pixel:, :]], axis=1)


        if height == 0:
            HEIGHTs_raw = ROWs_raw
            HEIGHTs_label = ROWs_label
        else:
            
            HEIGHTs_raw, ROWs_raw = overlap_fusing(HEIGHTs_raw, ROWs_raw, direction=1, halo_size=overloap_z_pixel,
                                           whether_label = False)

            HEIGHTs_raw = np.concatenate([HEIGHTs_raw, ROWs_raw[overloap_z_pixel:, :, :]], axis=0)

            
            HEIGHTs_label, ROWs_label = overlap_fusing(HEIGHTs_label, ROWs_label, direction=1, halo_size=overloap_z_pixel,
                                           whether_label = True)

            HEIGHTs_label = np.concatenate([HEIGHTs_label, ROWs_label[overloap_z_pixel:, :, :]], axis=0)

    HEIGHTs_label = np.uint32(HEIGHTs_label)

    save_raw_path = save_path + '/' + 'big_raw'
    save_label_path = save_path + '/' + 'big_label'
    save_color_path = save_path + '/' + 'colorful'
    if not os.path.exists(save_raw_path):
        os.mkdir(save_raw_path)
    if not os.path.exists(save_label_path):
        os.mkdir(save_label_path)
    if not os.path.exists(save_color_path):
        os.mkdir(save_color_path)

    whether_save_raw = True
    whether_save_label = True
    whether_save_colorful = True

    if whether_save_raw == True:

        names = os.listdir(save_raw_path)
        total = int(len(names))  
        if total < HEIGHTs_raw.shape[0]:
            print('begin write raw')
            for i in range(0, HEIGHTs_raw.shape[0]):
                imageio.imwrite(save_raw_path + '/' + str(i + 1).zfill(4) + '.tif', HEIGHTs_raw[i])
        else:
            print(save_raw_path, ' has already writed raw')

    if whether_save_label == True:

        names = os.listdir(save_label_path)
        total = int(len(names))  
        if total < HEIGHTs_label.shape[0]:
            print('begin write label')
            for i in range(0, HEIGHTs_label.shape[0]):
                imageio.imwrite(save_label_path + '/' + str(i + 1).zfill(4) + '.tif', HEIGHTs_label[i])
        else:
            print(save_label_path, ' has already writed label')

    if whether_save_colorful == True:

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
    sub_tifs.sort(key=lambda x: int(x.split('-')[0])*100 + int(x.split('_')[1])*10 + int(x.split('_')[2]) )

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
        arr_read = np.array(list(arr_read), dtype = np.int32)
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
    
    already_file_list = sorted(glob.glob(sub_data_path+'/*'))
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



if __name__ == '__main__':

 


    stack_names = os.listdir('/opt/data/Nas402/zbrfish/zbfStackResults_first_multicut_cc3d/')
    

    stack_names.sort(key=lambda x: int(x[5:]))


    outPutList_stack = []
    for stack in range(0, 15):   
        stack_name = stack_names[stack]
        print(stack_name)
        stack_path = '/opt/data/Nas402/zbrfish/zbfStackResults_first_multicut_cc3d/'
        matrix_path = '/opt/data/RecData/test/zbfStackResults_first_matrix/'
        save_path = '/opt/data/Nas402/zbrfish/zbf_merge_neuro/' + stack_name + '/'
        raw_path = '/opt/data/Nas402/zbrfish/zbfStackResults_first_enhance/' + stack_name + '/'

        
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        outPutList = get_processing_information(stack_name, stack_path, matrix_path, save_path, raw_path)
        if not outPutList == []:
            outPutList_stack.extend(outPutList)

    overloap_x_pixel = 600
    overloap_y_pixel = 600
    overloap_z_pixel = 1
    bite = 19
    cuda_num = 0
    width = 'all'
    affine_scale = 10 
    whether_update = False
    whether_save_breakpoint = False
    

    for x in outPutList_stack:
        merge_tif_bigdata_not_Regular_stack(x, overloap_x_pixel, overloap_y_pixel, overloap_z_pixel, bite, cuda_num, width, affine_scale, whether_update, whether_save_breakpoint)
        

        
        
        
        
        