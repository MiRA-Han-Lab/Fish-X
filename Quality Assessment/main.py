import cv2
import numpy as np
import math as m
import sys
# for gamma function, called 
from scipy.special import gamma as tgamma
import os
import argparse
import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2300000000
import glob
from multiprocessing import Pool, Lock, Value, Manager
path = "/home/liujz/bigstore/BRISQUE/libsvm/python/"
sys.path.append(path)

import svmutil
from svmutil import *
from svm import *

import matplotlib.pyplot as plt

#plt.ion()

# import svm functions (from libsvm library)
# if python2.x version : import svm from libsvm (sudo apt-get install python-libsvm)


# AGGD fit model, takes input as the MSCN Image / Pair-wise Product
def AGGDfit(structdis):
    # variables to count positive pixels / negative pixels and their squared sum
    poscount = 0
    negcount = 0
    possqsum = 0
    negsqsum = 0
    abssum   = 0

    poscount = len(structdis[structdis > 0]) # number of positive pixels
    negcount = len(structdis[structdis < 0]) # number of negative pixels
    
    # calculate squared sum of positive pixels and negative pixels
    possqsum = np.sum(np.power(structdis[structdis > 0], 2))
    negsqsum = np.sum(np.power(structdis[structdis < 0], 2))
    
    # absolute squared sum
    abssum = np.sum(structdis[structdis > 0]) + np.sum(-1 * structdis[structdis < 0])

    # calculate left sigma variance and right sigma variance
    lsigma_best = np.sqrt((negsqsum/negcount))
    rsigma_best = np.sqrt((possqsum/poscount))

    gammahat = lsigma_best/rsigma_best
    
    # total number of pixels - totalcount
    totalcount = structdis.shape[1] * structdis.shape[0]

    rhat = m.pow(abssum/totalcount, 2)/((negsqsum + possqsum)/totalcount)
    rhatnorm = rhat * (m.pow(gammahat, 3) + 1) * (gammahat + 1)/(m.pow(m.pow(gammahat, 2) + 1, 2))
    
    prevgamma = 0
    prevdiff  = 1e10
    sampling  = 0.001
    gam = 0.2

    # vectorized function call for best fitting parameters
    vectfunc = np.vectorize(func, otypes = [np.float], cache = False)
    
    # calculate best fit params
    gamma_best = vectfunc(gam, prevgamma, prevdiff, sampling, rhatnorm)

    return [lsigma_best, rsigma_best, gamma_best] 

def func(gam, prevgamma, prevdiff, sampling, rhatnorm):
    while(gam < 10):
        r_gam = tgamma(2/gam) * tgamma(2/gam) / (tgamma(1/gam) * tgamma(3/gam))
        diff = abs(r_gam - rhatnorm)
        if(diff > prevdiff): break
        prevdiff = diff
        prevgamma = gam
        gam += sampling
    gamma_best = prevgamma
    return gamma_best

def compute_features(img):
    scalenum = 2
    feat = []
    # make a copy of the image 
    im_original = img.copy()

    # scale the images twice 
    for itr_scale in range(scalenum):
        im = im_original.copy()
        # normalize the image
        im = im / 255.0

        # calculating MSCN coefficients
        mu = cv2.GaussianBlur(im, (7, 7), 1.166)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(im*im, (7, 7), 1.166)
        sigma = (sigma - mu_sq)**0.5
        
        # structdis is the MSCN image
        structdis = im - mu
        structdis /= (sigma + 1.0/255)
        
        # calculate best fitted parameters from MSCN image
        best_fit_params = AGGDfit(structdis)
        # unwrap the best fit parameters 
        lsigma_best = best_fit_params[0]
        rsigma_best = best_fit_params[1]
        gamma_best  = best_fit_params[2]
        
        # append the best fit parameters for MSCN image
        feat.append(gamma_best)
        feat.append((lsigma_best*lsigma_best + rsigma_best*rsigma_best)/2)

        # shifting indices for creating pair-wise products
        shifts = [[0,1], [1,0], [1,1], [-1,1]] # H V D1 D2

        for itr_shift in range(1, len(shifts) + 1):
            OrigArr = structdis
            reqshift = shifts[itr_shift-1] # shifting index

            # create transformation matrix for warpAffine function
            M = np.float32([[1, 0, reqshift[1]], [0, 1, reqshift[0]]])
            ShiftArr = cv2.warpAffine(OrigArr, M, (structdis.shape[1], structdis.shape[0]))
            
            Shifted_new_structdis = ShiftArr
            Shifted_new_structdis = Shifted_new_structdis * structdis
            # shifted_new_structdis is the pairwise product 
            # best fit the pairwise product 
            best_fit_params = AGGDfit(Shifted_new_structdis)
            lsigma_best = best_fit_params[0]
            rsigma_best = best_fit_params[1]
            gamma_best  = best_fit_params[2]

            constant = m.pow(tgamma(1/gamma_best), 0.5)/m.pow(tgamma(3/gamma_best), 0.5)
            meanparam = (rsigma_best - lsigma_best) * (tgamma(2/gamma_best)/tgamma(1/gamma_best)) * constant

            # append the best fit calculated parameters            
            feat.append(gamma_best) # gamma best
            feat.append(meanparam) # mean shape
            feat.append(m.pow(lsigma_best, 2)) # left variance square
            feat.append(m.pow(rsigma_best, 2)) # right variance square
        
        # resize the image on next iteration
        im_original = cv2.resize(im_original, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    return feat

# function to calculate BRISQUE quality score 
# takes input of the image path
def test_measure_BRISQUE(dis):
    # read image from given path
    #dis = cv2.imread(imgPath, -1)
    #if(dis is None):
    #    print("Wrong image path given")
    #    print("Exiting...")
    #    sys.exit(0)
    # convert to gray scale
    #dis = cv2.cvtColor(dis, cv2.COLOR_BGR2GRAY)

    # compute feature vectors of the image
    features = compute_features(dis)

    # rescale the brisqueFeatures vector from -1 to 1
    x = [0]
    
    # pre loaded lists from C++ Module to rescale brisquefeatures vector to [-1, 1]
    min_= [0.336999 ,0.019667 ,0.230000 ,-0.125959 ,0.000167 ,0.000616 ,0.231000 ,-0.125873 ,0.000165 ,0.000600 ,0.241000 ,-0.128814 ,0.000179 ,0.000386 ,0.243000 ,-0.133080 ,0.000182 ,0.000421 ,0.436998 ,0.016929 ,0.247000 ,-0.200231 ,0.000104 ,0.000834 ,0.257000 ,-0.200017 ,0.000112 ,0.000876 ,0.257000 ,-0.155072 ,0.000112 ,0.000356 ,0.258000 ,-0.154374 ,0.000117 ,0.000351]
    
    max_= [9.999411, 0.807472, 1.644021, 0.202917, 0.712384, 0.468672, 1.644021, 0.169548, 0.713132, 0.467896, 1.553016, 0.101368, 0.687324, 0.533087, 1.554016, 0.101000, 0.689177, 0.533133, 3.639918, 0.800955, 1.096995, 0.175286, 0.755547, 0.399270, 1.095995, 0.155928, 0.751488, 0.402398, 1.041992, 0.093209, 0.623516, 0.532925, 1.042992, 0.093714, 0.621958, 0.534484]

    # append the rescaled vector to x 
    for i in range(0, 36):
        min = min_[i]
        max = max_[i] 
        x.append(-1 + (2.0/(max - min) * (features[i] - min)))
    
    # load model 
    model = svmutil.svm_load_model("allmodel")

    # create svm node array from python list
    x, idx = gen_svm_nodearray(x[1:], isKernel=(model.param.kernel_type == PRECOMPUTED))
    x[36].index = -1 # set last index to -1 to indicate the end.
	
	# get important parameters from model
    svm_type = model.get_svm_type()
    is_prob_model = model.is_probability_model()
    nr_class = model.get_nr_class()
    
    if svm_type in (ONE_CLASS, EPSILON_SVR, NU_SVC):
        # here svm_type is EPSILON_SVR as it's regression problem
        nr_classifier = 1
    dec_values = (c_double * nr_classifier)()
    
    # calculate the quality score of the image using the model and svm_node_array
    qualityscore = svmutil.libsvm.svm_predict_probability(model, x, dec_values)

    return qualityscore

def brisque(image_dir):
    qualityscore = test_measure_BRISQUE(image_dir)
    return qualityscore

def get_location(big_image, x_time, y_time, row, start_row, end_row, line, start_line, end_line, random_size_kuan, random_size_chang):


    y = [0, big_image.shape[0]]
    x = [0, big_image.shape[1]]

    if row == start_row:
        y[0] = int(y[1] / 3 * 2)
    elif row == end_row:
        y[1] = int(y[1] - (y[1] - y[0]) / 3 * 2)

    if line == start_line:
        x[0] = int(x[1] / 3 * 2)

    elif line == end_line:
        x[1] = int(x[1] - (x[1] - x[0]) / 3 * 2)

    x1 = [0, 0]
    y1 = [0, 0]

    if x_time == 1:
        x1[0] = x[0]
        x1[1] = int((x[1] - x[0]) / 2 + x[0])
    else:
        x1[0] = int((x[1] - x[0]) / 2 + x[0])
        x1[1] = x[1]

    if y_time == 1:
        y1[0] = y[0]
        y1[1] = int((y[1] - y[0]) / 2 + y[0])
    else:
        y1[0] = int((y[1] - y[0]) / 2 + y[0])
        y1[1] = y[1]

    choosen_kuan = np.random.randint(y1[0], y1[1] - random_size_kuan)
    choosen_chang = np.random.randint(x1[0], x1[1] - random_size_chang)


    #image_part = np.array(big_image.crop([choosen_chang, choosen_kuan, choosen_chang + random_size_chang, choosen_kuan + random_size_kuan]))

    image_part = big_image[choosen_kuan:choosen_kuan + random_size_kuan, choosen_chang:choosen_chang + random_size_chang]

    return image_part


def test_good_images(refer_good_path, start_row, end_row, start_line, end_line, random_size_kuan, random_size_chang):
    ################leran reference sorce
    ################good###################
    refer_filenames = os.listdir(refer_good_path)
    refer_sorce_good = []
    for name in refer_filenames:
        path = refer_good_path + '/' + name
        qualityscore = 0
        all = name.split('.')[0]
        row = int(all.split('_')[-2])
        line = int(all.split('_')[-1])
        refer_good = []
        for x_time in range(1, 3):
            for y_time in range(1, 3):
                for time in range(5):
                    image_part = get_location(path, x_time, y_time, row, start_row, end_row, line, start_line, end_line, random_size_kuan, random_size_chang)
                    plt.imshow(image_part, cmap='gray')

                    qualityscore = test_measure_BRISQUE(image_part)
                    refer_good.append(qualityscore)

                    print("Score of the good reference " + name + ' ' + str(((x_time-1)*10+(y_time-1)*5)+(time+1)) + 'times' + ': ', qualityscore)

        refer_good.remove(max(refer_good))
        refer_good.remove(max(refer_good))
        refer_good.remove(min(refer_good))
        refer_good.remove(min(refer_good))
        mean_qualityscore = np.mean(refer_good)

        refer_sorce_good.append(mean_qualityscore)

    mean_good_qualityscore = np.mean(refer_sorce_good)
    #median_good_qualityscore = np.median(refer_sorce_good)
    print("Mean good qualityscore of the reference ", mean_good_qualityscore)
    #print("Median good qualityscore of the reference ", median_good_qualityscore)
    return mean_good_qualityscore



def test_bad_images(refer_bad_path, start_row, end_row, start_line, end_line, random_size_kuan, random_size_chang):
    #################leran reference sorce
    #################bad###################
    refer_filenames = os.listdir(refer_bad_path)
    refer_sorce_bad = []
    for name in refer_filenames:
        path = refer_bad_path + '/' + name
        qualityscore = 0
        all = name.split('.')[0]
        row = int(all.split('_')[-2])
        line = int(all.split('_')[-1])
        refer_bad = []
        for x_time in range(1, 3):
            for y_time in range(1, 3):
                for time in range(5):
                    image_part = get_location(path, x_time, y_time, row, start_row, end_row, line, start_line, end_line,
                                              random_size_kuan, random_size_chang)
                    plt.imshow(image_part, cmap='gray')
                    qualityscore = test_measure_BRISQUE(image_part)
                    refer_bad.append(qualityscore)

                    print("Score of the bad reference " + name + ' ' + str(
                        ((x_time - 1) * 10 + (y_time - 1) * 5) + (time + 1)) + ' times' + ': ', qualityscore)

        refer_bad.remove(max(refer_bad))
        refer_bad.remove(max(refer_bad))
        refer_bad.remove(min(refer_bad))
        refer_bad.remove(min(refer_bad))
        mean_qualityscore = np.mean(refer_bad)

        refer_sorce_bad.append(mean_qualityscore)

        # if mean_qualityscore > 60:
        #     with open(bad_txt_path, 'a', encoding='utf-8') as f:
        #         text = test_path + '/' + path  + '\n'
        #         f.write(text)

    mean_bad_qualityscore = np.mean(refer_sorce_bad)
    #median_bad_qualityscore = np.median(refer_sorce_bad)
    print("Mean bad qualityscore of the reference ", mean_bad_qualityscore)
    #print("Median bad qualityscore of the reference ", median_bad_qualityscore)
    return mean_bad_qualityscore

def sort_already_image(already_txt_path, sort_txt_path):
    empty_dict = dict()
    with open(already_txt_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            name = line.rsplit(" ", 1)[0]
            score = float(line.rsplit(" ", 1)[1])
            empty_dict[name] = score

    empty_dict = sorted(empty_dict.items(), key=lambda d: d[1], reverse=True)

    with open(sort_txt_path, 'a', encoding='utf-8') as f:
        for line in empty_dict:
            name = line[0]
            score = line[1]
            text = name + ' ' + str(score) + '\n'
            f.write(text)

    return empty_dict


def test_all_data(parameter_list):
    #################cal test sorce

    for path in parameter_list:
        qualityscore = 0
        images = []
        name = path.rsplit(".", 1)[0]

        row = int(name.split('_')[-2])
        line = int(name.split('_')[-1])
        big_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        for x_time in range(1, 3):
            for y_time in range(1, 3):
                for time in range(resample):

                    image_part = get_location(big_image, x_time, y_time, row, start_row, end_row, line, start_line,
                                              end_line, random_size_kuan, random_size_chang)
                    plt.imshow(image_part, cmap='gray')
                    gradient, var = calcul_gradient_var(image_part)
                    count = 0
                    while ((whether_white_area(gradient, var, threshold_gradient, threshold_var) == False) and (
                            count < 3)):
                        image_part = get_location(big_image, x_time, y_time, row, start_row, end_row, line, start_line,
                                                  end_line, random_size_kuan, random_size_chang)
                        gradient, var = calcul_gradient_var(image_part)
                        count = count + 1

                    qualityscore = test_measure_BRISQUE(image_part)
                    images.append(qualityscore)

        images.remove(max(images))
        images.remove(max(images))
        images.remove(min(images))
        images.remove(min(images))
        mean_qualityscore = sum(images) / 8
        print("Score of the given layer " + path, " image: ", mean_qualityscore)

        # if qualityscore < Threshold_qualityscore:
        #     pass
        #
        #
        # elif qualityscore < (1 + relax_m) * Threshold_qualityscore:
        #     haixingname = '33_444' + '_' + str(layer).zfill(3) + '_' + str(row) + '_' + str(line)
        #     haixing_images.append(haixingname)
        #     with open(haixing_txt_path, 'a', encoding='utf-8') as f:
        #         text = test_path + '/' + haixingname + '.tif' + '\n'
        #         f.write(text)
        #     print("possible-conformity product: ", haixingname)
        #
        # else:
        #     badname = '33_444' + '_' + str(layer).zfill(3) + '_' + str(row) + '_' + str(line)
        #     bad_images.append(badname)
        #     with open(bad_txt_path, 'a', encoding='utf-8') as f:
        #         text = test_path + '/' + badname + '.tif' + '\n'
        #         f.write(text)
        #     print("non-conformity product: ", badname)
        with open(already_txt_path, 'a', encoding='utf-8') as f:
            text = path + ' ' + str(mean_qualityscore) + '\n'
            f.write(text)

def procerss_test_all_data(path):

    all = path.rsplit(".", 1)[0]
    row = int(all.split('_')[-2])
    line = int(all.split('_')[-1])
    qualityscore = 0
    images = []
    big_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    big_image = cv2.equalizeHist(big_image)
    #cv2.imshow("dst", big_image)

    for x_time in range(1, 3):
        for y_time in range(1, 3):
            for time in range(resample):
                image_part = get_location(big_image, x_time, y_time, row, start_row, end_row, line, start_line,
                                          end_line, random_size_kuan, random_size_chang)
                #plt.imshow(image_part, cmap='gray')
                gradient, var = calcul_gradient_var(image_part)
                count=0
                while ((whether_white_area(gradient, var, threshold_gradient, threshold_var) == False) and (count < 3)):
                    image_part = get_location(big_image, x_time, y_time, row, start_row, end_row, line, start_line,
                                              end_line, random_size_kuan, random_size_chang)
                    gradient, var = calcul_gradient_var(image_part)
                    count = count + 1

                qualityscore = test_measure_BRISQUE(image_part)
                images.append(qualityscore)

    images.remove(max(images))
    images.remove(max(images))
    images.remove(min(images))
    images.remove(min(images))
    mean_qualityscore = sum(images) / 8
    print(path," 's score: ", mean_qualityscore)

    lock.acquire()  ########lock
    with open(already_txt_path, 'a', encoding='utf-8') as f:
        text = path + ' ' + str(mean_qualityscore) + '\n'
        f.write(text)
    lock.release()  ########lock

    # if mean_qualityscore < Threshold_qualityscore:
    #     goodname = path
    #     # lock.acquire()  ########lock
    #     with open(good_txt_path, 'a', encoding='utf-8') as f:
    #         text = goodname + '\n'
    #         f.write(text)
    #     # lock.release()  ########lock
    #     print("good product: ", goodname)
    #
    #
    # elif mean_qualityscore < (1 + relax_m) * Threshold_qualityscore:
    #     haixingname = path
    #     #lock.acquire()  ########lock
    #     with open(haixing_txt_path, 'a', encoding='utf-8') as f:
    #         text = haixingname + '\n'
    #         f.write(text)
    #     #lock.release()  ########lock
    #     print("possible-conformity product: ", haixingname)
    #
    # else:
    #     badname = path
    #     #lock.acquire()  ########lock
    #     with open(bad_txt_path, 'a', encoding='utf-8') as f:
    #         text = badname + '\n'
    #         f.write(text)
    #     #lock.release()  ########lock
    #     print("non-conformity product: ", badname)

def procerss_test_all_data2(path):
    print(path)
    big_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("dst", big_image)
    big_image = cv2.equalizeHist(big_image)
    cv2.imshow("dst", big_image)
def Progress_check(already_txt_path, all_path):
    # good_list=[]
    # bad_list=[]
    # haixing_list=[]
    # with open(good_path, "r") as f:
    #     for line in f.readlines():
    #         line = line.strip('\n')  # 去掉列表中每一个元素的换行符
    #         good_list.append(line)
    #         #print(line, ' has been detected')
    # with open(bad_path, "r") as f:
    #     for line in f.readlines():
    #         line = line.strip('\n')  # 去掉列表中每一个元素的换行符
    #         bad_list.append(line)
    #         #print(line, ' has been detected')
    # with open(haixing_path, "r") as f:
    #     for line in f.readlines():
    #         line = line.strip('\n')  # 去掉列表中每一个元素的换行符
    #         haixing_list.append(line)
    #         #print(line, ' has been detected')


    all_data_path = glob.glob(all_path + '/' + '*.tif')

    print('All data obtained')
    # for line in good_list:
    #     if line in all_data_path:
    #         all_data_path.remove(line)
    # for line in bad_list:
    #     if line in all_data_path:
    #         all_data_path.remove(line)
    # for line in haixing_list:
    #     if line in all_data_path:
    #         all_data_path.remove(line)

    already_list=[]
    with open(already_txt_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            name = line.rsplit(" ", 1)[0]
            #score = line.rsplit(" ", 1)[1]
            already_list.append(name)

    for line in already_list:
        if line in all_data_path:
            all_data_path.remove(line)

    print('Successful resuming of breakpoint')

    return all_data_path

def compare_already_image():
    a=1


def calcul_gradient_var(image_part):
    ##############################图像差分值计算################

    image_part_Gaussian = cv2.GaussianBlur(image_part, (7, 7), 0)

    scharrx = cv2.Scharr(image_part_Gaussian, cv2.CV_64F, dx=1, dy=0)
    #scharrx = cv2.convertScaleAbs(scharrx)
    scharry = cv2.Scharr(image_part_Gaussian, cv2.CV_64F, dx=0, dy=1)
    #scharry = cv2.convertScaleAbs(scharry)
    result = cv2.addWeighted(abs(scharrx), 0.5, abs(scharry), 0.5, 0)
    result_sq = cv2.addWeighted((scharrx*scharrx), 0.5, (scharry*scharry), 0.5, 0)

    gradient_sum = sum(sum(result))/100000
    gradient_sq_sum = sum(sum(result_sq))/100000

    var_sum = cv2.Laplacian(image_part_Gaussian, cv2.CV_64F).var()

    return gradient_sum, var_sum

def whether_white_area(gradient_sum, var_sum, threshold_gradient, threshold_var):

    if (gradient_sum < threshold_gradient) and (var_sum < threshold_var):
        return False
    else:
        return True


def learn_white_area(path_white, path_normal):
    #################leran gradient and var##############
    #################white###############################
    refer_white_filenames = os.listdir(path_white)
    refer_gradient_white = []
    refer_var_white = []
    for name in refer_white_filenames:
        path = path_white + '/' + name
        #all = name.split('.')[0]
        #row = int(all.split('_')[-2])
        #line = int(all.split('_')[-1])
        gradient_white = []
        var_white = []
        for x_time in range(1, 3):
            for y_time in range(1, 3):
                for time in range(5):
                    image_part = get_location(path, x_time, y_time, 1, start_row, end_row, 1, start_line, end_line,
                                              random_size_kuan, random_size_chang)
                    plt.imshow(image_part, cmap='gray')
                    gradient_sum, var_sum = calcul_gradient_var(image_part)

                    gradient_white.append(gradient_sum)
                    var_white.append(var_sum)

        gradient_white.remove(max(gradient_white))
        gradient_white.remove(max(gradient_white))
        gradient_white.remove(min(gradient_white))
        gradient_white.remove(min(gradient_white))
        var_white.remove(max(var_white))
        var_white.remove(max(var_white))
        var_white.remove(min(var_white))
        var_white.remove(min(var_white))

        gradient = np.mean(gradient_white)
        var = np.mean(var_white)

        print('Gradient of the white image: ', name, 'is', gradient)
        print('Var of the white image: ', name, 'is', var)

        refer_gradient_white.append(gradient)
        refer_var_white.append(var)

    mean_gradient_white = np.mean(refer_gradient_white)
    mean_var_white = np.mean(refer_var_white)

    print("Mean gradient of the white ", mean_gradient_white)
    print("Mean var of the white ", mean_var_white)

    #################normal###############################
    refer_normal_filenames = os.listdir(path_normal)
    refer_gradient_normal = []
    refer_var_normal = []
    for name in refer_normal_filenames:
        path = path_normal + '/' + name
        # all = name.split('.')[0]
        # row = int(all.split('_')[-2])
        # line = int(all.split('_')[-1])
        gradient_normal = []
        var_normal = []
        for x_time in range(1, 3):
            for y_time in range(1, 3):
                for time in range(5):
                    image_part = get_location(path, x_time, y_time, 1, start_row, end_row, 1, start_line, end_line,
                                              random_size_kuan, random_size_chang)
                    plt.imshow(image_part, cmap='gray')
                    gradient_sum, var_sum = calcul_gradient_var(image_part)

                    gradient_normal.append(gradient_sum)
                    var_normal.append(var_sum)

        gradient_normal.remove(max(gradient_normal))
        gradient_normal.remove(max(gradient_normal))
        gradient_normal.remove(min(gradient_normal))
        gradient_normal.remove(min(gradient_normal))
        var_normal.remove(max(var_normal))
        var_normal.remove(max(var_normal))
        var_normal.remove(min(var_normal))
        var_normal.remove(min(var_normal))

        gradient = np.mean(gradient_normal)
        var = np.mean(var_normal)

        print('Gradient of the normal image: ', name, 'is', gradient)
        print('Var of the normal image: ', name, 'is', var)

        refer_gradient_normal.append(gradient)
        refer_var_normal.append(var)

    mean_gradient_normal = np.mean(refer_gradient_normal)
    mean_var_normal = np.mean(refer_var_normal)

    print("Mean gradient of the normal ", mean_gradient_normal)
    print("Mean var of the normal ", mean_var_normal)


    return mean_gradient_white, mean_var_white, mean_gradient_normal, mean_var_normal


# calculate quality score    #
#/home/liujz/3dian4/Image Dataset by FBT/DuJiulin/5#/whole brain/wafer34   ok/wafer34_4nm_421_85ns_7x8
#/home/liujz/3dian226/DataPro.semat/sem.gukiesoft.dujl.zebrafish/region35.12k.4.0nm.221.Wafer35
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EM Image Quality Assessment ')
    parser.add_argument('--test_path', '-t1', type=str, default='/home/liujz/3dian226/DataPro.semat/sem.gukiesoft.dujl.zebrafish/zebrafish-wafer15/region15.12k.4.0nm.221.Wafer15', help='source_path')
    parser.add_argument('--refer_bad_path', '-t2', type=str, default='/home/liujz/3dian246/DataPro.semat/sem.gukiesoft.dujl.zebrafish/wafer33-bad2-copy', help='refer bad path')
    parser.add_argument('--refer_good_path', '-t3', type=str, default='/home/liujz/bigstore/SCN/good_banmayu', help='refer good path')
    parser.add_argument('--path_white', '-p1', type=str, default='/home/liujz/bigstore/SCN/path_white', help='path_white')
    parser.add_argument('--path_normal', '-p2', type=str, default='/home/liujz/bigstore/SCN/path_normal', help='path_normal')
    parser.add_argument('--random_size_kuan', '-k1', type=int, default='1000', help='random_size_kuan')
    parser.add_argument('--random_size_chang', '-k2', type=int, default='1000', help='random_size_chang')
    parser.add_argument('--start_layer', '-s1', type=int, default='0', help='start_layer')
    parser.add_argument('--start_row', '-s2', type=int, default='0', help='start_row')
    parser.add_argument('--start_line', '-s3', type=int, default='0', help='start_line')
    parser.add_argument('--end_layer', '-e1', type=int, default='403', help='end_layer')
    parser.add_argument('--end_row', '-e2', type=int, default='5', help='end_row')
    parser.add_argument('--end_line', '-e3', type=int, default='6', help='end_line')
    parser.add_argument('--relax_m', '-m1', type=float, default='0.15', help='relax_factor')
    parser.add_argument('--resample', '-r', type=int, default='3', help='Resampling time')
    #parser.add_argument('--bad_txt_path', '-text1', type=str, default='/home/liujz/bigstore/SCN/banmayu/bad.txt', help='saved bad txt path')
    #parser.add_argument('--haixing_txt_path', '-text2', type=str, default='/home/liujz/bigstore/SCN/banmayu/haixing.txt', help='saved haixing txt path')
    #parser.add_argument('--good_txt_path', '-text3', type=str, default='/home/liujz/bigstore/SCN/banmayu/good.txt', help='saved good txt path')
    parser.add_argument('--already_txt_path', '-text4', type=str, default='/home/liujz/bigstore/SCN/banmayu/all_text_wafer10.txt', help='saved all txt path')
    parser.add_argument('--sort_txt_path', '-text5', type=str, default='/home/liujz/bigstore/SCN/banmayu/sort_wafer10.txt', help='sort txt path')


    args = parser.parse_args()

    test_path = args.test_path
    refer_bad_path = args.refer_bad_path
    refer_good_path = args.refer_good_path
    path_white = args.path_white
    path_normal = args.path_normal
    global start_layer
    start_layer = args.start_layer
    global start_row
    start_row = args.start_row
    global start_line
    start_line = args.start_line
    global end_layer
    end_layer = args.end_layer
    global end_row
    end_row = args.end_row
    global end_line
    end_line= args.end_line
    global random_size_kuan
    random_size_kuan = args.random_size_kuan
    global random_size_chang
    random_size_chang = args.random_size_chang
    global relax_m
    relax_m = args.relax_m
    global resample
    resample = args.resample
    #global bad_txt_path
    #bad_txt_path = args.bad_txt_path
    #global haixing_txt_path
    #haixing_txt_path = args.haixing_txt_path
    #global good_txt_path
    #good_txt_path = args.good_txt_path
    global already_txt_path
    already_txt_path = args.already_txt_path
    global sort_txt_path
    sort_txt_path = args.sort_txt_path

    Whether_Train = False


    global Threshold_qualityscore


    if Whether_Train == True:
        mean_good_qualityscore = test_good_images(refer_good_path, start_row, end_row, start_line, end_line, random_size_kuan, random_size_chang)
        mean_bad_qualityscore = test_bad_images(refer_bad_path, start_row, end_row, start_line, end_line, random_size_kuan, random_size_chang)
        Threshold_qualityscore = (mean_good_qualityscore + mean_bad_qualityscore)/2
    else:
        Threshold_qualityscore = 53

    global threshold_gradient
    global threshold_var

    #mean_gradient_white, mean_var_white, mean_gradient_normal, mean_var_normal = learn_white_area(path_white, path_normal)
    #threshold_gradient = (mean_gradient_white + mean_gradient_normal)/2
    #threshold_var = (mean_var_white + mean_var_normal)/2
    #558   8   1582   23

    threshold_gradient=1050
    threshold_var=14

    #procerss_test_all_data2('/home/liujz/3dian242/DataPro.semat/sem.gukiesoft.dujl.zebrafish/region18.12k.4.0nm.221.Wafer18/18_439_410_5_4.tif')

    all_data_path = Progress_check(already_txt_path, test_path)

    #test_all_data(all_data_path)

    parameter_list = all_data_path
    lock = Lock()
    pool = Pool(10, initargs=(lock,))
    pool.map(procerss_test_all_data, parameter_list)  # 表示将 sh_list 每个元素作为参数递给 run_sh
    pool.close()  # 将进程池关闭，不再接受新的进程
    pool.join()  # 主进程阻塞，只有池中所有进程都完毕了才会通过

    #sorted_dict = sort_already_image(already_txt_path, sort_txt_path)#####sort
    print('over')






    #test_all_data(test_path, start_row, end_row, start_line, end_line, random_size_kuan, random_size_chang, Threshold_qualityscore, relax_m, bad_txt_path, haixing_txt_path)
    # parameter_list=[]
    # for layer in range(start_layer, end_layer + 1):
    #     for row in range(start_row, end_row + 1):
    #         for line in range(start_line, end_line + 1):
    #             #parameter_list.append((layer, row, line, start_row, end_row, start_line, end_line, random_size_kuan,
    #                        #random_size_chang, Threshold_qualityscore, relax_m, bad_txt_path, haixing_txt_path))
    #             path = test_path + '/' + '33_444' + '_' + str(layer).zfill(3) + '_' + str(row) + '_' + str(
    #                 line) + '.tif'
    #             parameter_list.append(path)





