import numpy as np
import imageio

# import napari for data visualisation
# import napari
# import the segmentation functionality from elf
import elf.segmentation.multicut as mc
import elf.segmentation.features as feats
import elf.segmentation.watershed as ws
from elf.segmentation.utils import seg_to_edges
from elf.segmentation.features import *
from elf.segmentation.learning import *
import os,glob
import h5py
import scipy.io as sio
import joblib
from nifty import tools as ntools
import nifty
from scipy import ndimage as ndi
from skimage import feature, morphology, filters
from skimage.morphology import remove_small_objects, dilation, opening, closing
from scipy.ndimage import binary_fill_holes
import time
from tqdm import tqdm
from scipy.ndimage import affine_transform
import cv2
#from pyspark import SparkContext
#from pyspark import SparkConf
#import pickle

used_threads = 15

from PIL import Image
Image.MAX_IMAGE_PIXELS = 2300000000


os.environ['PYTHONPATH'] = '/opt/anaconda3/bin/python3'
os.environ['PYTHONHOME'] = '/opt/anaconda3/'
os.environ['PYSPARK_PYTHON'] = '/opt/anaconda3/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/opt/anaconda3/bin/python3'
os.environ['SPARK_HOME'] = '/opt/spark'


def read_chen_z_y_x(filename, mode):
    if mode == 'test':
        if len(filename.split('.')) == 1:
            data = read_32_tif_seqqence(filename)
            print('reading over')
        elif len((filename.split('.')[0].split('/')[-1]).split('_')) == 1:
            suffix = filename.split(".")[1]
            filename_text = filename.split(".")[0]
            name = (filename_text.rsplit("/", 1)[1])
            address = (filename_text.rsplit("/", 1)[0])

            order_start = name.split("-")[0]
            order_end = name.split("-")[1]
            order_layer = int(order_end) - int(order_start) + 1

            example = cv2.imread(address + '/' + str(order_start).zfill(4) + '.' + suffix, cv2.IMREAD_GRAYSCALE)
            
            print('example: ' + address + '/' + str(order_start).zfill(4) + '.' + suffix)

            data = np.zeros((int(order_layer), example.shape[0], example.shape[1]), dtype=np.uint8)

            del example

            for i in range(int(order_layer)):
                image_name = address + '/' + str(i + int(order_start)).zfill(4) + '.' + suffix  # image_type
                image_temp = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

                data[i, :, :] = image_temp
            print('reading over')

        else:
            suffix = filename.split(".")[1]
            filename_text = filename.split(".")[0]

            name = (filename_text.rsplit("/", 1)[1])
            address = (filename_text.rsplit("/", 1)[0])
            order1 = (name.split("_")[1])

            order_start = order1.split("-")[0]
            order_end = order1.split("-")[1]
            order_layer = int(order_end) - int(order_start) + 1
            # order_layer = re.findall("\d+", order1)[0]  # 得到字符串中的数�?
            order_y = (filename_text.split("_")[-2])
            order_x = (filename_text.split("_")[-1])

            example = cv2.imread(
                address + '/' + 'layer' + str(order_start) + '_' + str(order_y) + '_' + str(order_x) + '.' + suffix,
                cv2.IMREAD_GRAYSCALE)
                
            #if example == None:
            #        f = open('/opt/data/Nas402/zbrfish/liujz_code' + '/' + 'eerror.txt', mode='a')
            #        f.write('error:'+ '\t' + address + '/' + 'layer' + str(order_start) + '_' + str(order_y) + '_' + str(order_x) + '.' + suffix + '\n')  # write 写入
            #        f.close()  # 关闭文件

            data = np.zeros((int(order_layer), example.shape[0], example.shape[1]), dtype=np.uint8)

            del example

            for i in range(int(order_layer)):
                image_name = address + '/' + 'layer' + str(i + int(order_start)) + '_' + str(order_y) + '_' + str(
                    order_x) + '.' + suffix  # image_type
                image_temp = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

                data[i, :, :] = image_temp
            print('reading over')

    if mode == 'train':
        data = read_32_tif_seqqence(filename)
        print('reading over')

    return data


def Progress_check_label(already_path):

    sub_data_path = os.path.join(already_path)
    if not os.path.exists(sub_data_path):
         os.makedirs(sub_data_path)
    #already_file_list = os.listdir(sub_data_path)
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


def get_affinity_information(load_path):
    #######################################
    #all_data_path = glob.glob(load_path + '/*.h5')

    read_suffix = '.jpg'

    all_data_path = glob.glob(load_path + '/' + '*' + read_suffix)


    def get_1(c):
        return c.split('/')[-1]

    def get_2(c):
        return c.split('.')[0]

    sub_tifs = [get_1(x) for x in all_data_path]
    sub_tifs = [get_2(x) for x in sub_tifs]
    # print('All data obtained')
    total = int(len(sub_tifs))
    # sub_tifs.sort(
    #     key=lambda x: (int(x[5:].split('_')[0]) * 10000 + int(x[5:].split('_')[1]) * 100 + int(x[5:].split('_')[2])))
    sorted_file = sub_tifs

    # 璁＄畻num_x,num_y,num_z
    height_all = np.zeros(total)
    row_all = np.zeros(total)
    line_all = np.zeros(total)

    for i, name in enumerate(sorted_file):
        height_all[i] = int(name[5:].split('_')[0])
    for i, name in enumerate(sorted_file):
        row_all[i] = int(name[5:].split('_')[1])
    for i, name in enumerate(sorted_file):
        line_all[i] = int(name[5:].split('_')[2])

    number_z = np.unique(height_all)
    number_y = np.unique(row_all)
    number_x = np.unique(line_all)

    num_z = len(number_z)
    num_y = len(number_y)
    num_x = len(number_x)

    FilenameAray = []
    for j in range(num_z):
        FilenameAray_xy = []
        for i in range(num_y):
            zeroArray = ['0' for i in range(num_x)]
            FilenameAray_xy.append(zeroArray)
        FilenameAray.append(FilenameAray_xy)

    #print('row_all', row_all)
    #print('line_all', line_all)
    #print('height_all', height_all)


    min_y = int(np.min(row_all))
    max_y = int(np.max(row_all))
    min_x = int(np.min(line_all))
    max_x = int(np.max(line_all))
    #min_z = int(sorted_file[0].split('_')[1].split('-')[0])
    #max_z = int(sorted_file[0].split('_')[1].split('-')[1])
    min_z = int(np.min(height_all))
    max_z = int(np.max(height_all))

    del row_all, line_all, height_all, FilenameAray_xy, number_y, number_x, number_z

    return min_z, max_z, min_y, max_y, min_x, max_x


def get_processing_information(stack_name, img_path, pre_path, matrix_path, save_path):
    ##########################
    #### get already list ####
    already_list = Progress_check_label(save_path+'/'+stack_name)
    ##########################
    load_path = img_path+stack_name
    #### get already list ####
    min_z, max_z, min_y, max_y, min_x, max_x = get_affinity_information(load_path)
    
    #min_y = 3
    #max_y = 6
    #min_x = 8
    #max_x = 11
    
    outPutList = []
    outPut = []
    for ii in range(min_y, max_y+1):
        for jj in range(min_x, max_x+1):
            IMAGE_NAME = img_path + stack_name + '/layer' + '_' + str(min_z) + '-' + \
                         str(max_z) + '_' + str(ii) + '_' + str(jj) + '.jpg'
            PRE_NAME = pre_path + stack_name + '/' + str(min_z).zfill(5) + '-' + str(max_z).zfill(5) + '_' + \
                          str(ii).zfill(2) + '_' + str(jj).zfill(2) #+ '.h5'
            #Aff_name = matrix_path + stack_name + '/affine_' + str(ii).zfill(2) + str(jj).zfill(2) +'.mat'
            Aff_name = matrix_path + '/'

            #OUTPUT_PATH1 = save_path1 + stack_name + '/' + str(min_z).zfill(5) + '-' + str(max_z).zfill(5) + '_' + str(ii).zfill(2).zfill(2) + '_' + str(jj).zfill(2)
            OUTPUT_PATH = save_path + stack_name + '/' + str(min_z).zfill(5) + '-' + str(max_z).zfill(5) + '_' + str(ii).zfill(2).zfill(2) + '_' + str(jj).zfill(2)
            if str(min_z).zfill(5) + '-' + str(max_z).zfill(5) + '_' + str(ii).zfill(2).zfill(2) + '_' + str(jj).zfill(2) in already_list:
                continue
                
            #have_list = ['00001-00031_02_09']
            #if str(min_z).zfill(5) + '-' + str(max_z).zfill(5) + '_' + str(ii).zfill(2).zfill(2) + '_' + str(jj).zfill(2) not in have_list:
            #    continue    
            
            outPut.append(IMAGE_NAME)
            outPut.append(PRE_NAME)
            outPut.append(Aff_name)
            #outPut.append(OUTPUT_PATH1)
            outPut.append(OUTPUT_PATH)
            outPutList.append(outPut)
            outPut = []

    return outPutList


def compute_matrix(matrix_cell, i):
    if i == 1:
        matrix = matrix_cell[i - 1]
    else:
        matrix = np.linalg.multi_dot(matrix_cell[0:i])
    return np.linalg.inv(matrix)


def affine_trans(raw, filename, affineDataPath, affine_scale, Whether_invert = False):

    if len(filename.split('.')) == 1:
        name = filename
    elif len(filename.split('.')) == 2:
        name = filename.split('.')[0]
        save_suffix = filename.split('.')[1]

    order_x = (name.split("_")[-2])
    order_y = (name.split("_")[-1])
    order_stack = name.rsplit("/")[-2]
    #matrix_path = affineDataPath + "/" + order_stack + '/affine_' + str(order_x).zfill(2) + str(order_y).zfill(2) +'.mat'
    matrix_path = affineDataPath + "/" + order_stack + '/affine_' + str(order_x).zfill(2) + str(order_y).zfill(2) +'.mat'
    matrix_cell = sio.loadmat(matrix_path)['affine'][0]
    matrix_cell = matrix_cell[-raw.shape[0] + 1:]
    for i in range(raw.shape[0] - 1):
        ## resize factor is 10
        matrix_cell[i][0, 2] = matrix_cell[i][0, 2] * affine_scale #* 120/128
        matrix_cell[i][1, 2] = matrix_cell[i][1, 2] * affine_scale #* 120/128
        
        ######################
        error_index = matrix_cell[i][0, 0] * matrix_cell[i][1, 1] - matrix_cell[i][0, 1] * matrix_cell[i][1, 0]
        if error_index > 1.15 or error_index < 0.85:
            matrix_cell[i] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        ######################

    out = np.zeros_like(raw)
    for i in range(raw.shape[0]):
        im = raw[i].T
        order = 0   #########  0 is label,1 is chazhi
        if i == 0:
            out[i] = im.T
        else:
            matrix = compute_matrix(matrix_cell, i)
            if Whether_invert == True:
                matrix = np.linalg.inv(matrix)
            out[i] = affine_transform(im, matrix, order=order).T
    return out





def compute_feature(watershed, raw, boundaries):
    rag = feats.compute_rag(watershed)

    t0 = time.time()
    feature1 = compute_boundary_features_with_filters(rag, raw.astype('float32'), apply_2d=True, n_threads = used_threads)
    print('compute_boundary_features_with_filters takes time:', time.time() - t0)

    t1 = time.time()
    feature2 = compute_boundary_features(rag, boundaries, n_threads = used_threads)
    print('compute_boundary_features takes time:', time.time() - t1)

    t2 = time.time()
    feature3 = compute_region_features(rag.uvIds(), raw.astype('float32'), watershed.astype('uint32'), n_threads = used_threads)
    print('compute_region_features takes time:', time.time() - t2)

    features = np.column_stack([feature1, feature2, feature3])
    return features, rag

def learning_edge(watershed, raw_train, boundaries_train, gtImage, save_path='saved_model'):
    # # start learning
    features, rag = compute_feature(watershed, raw_train, boundaries_train)
    edge_labels  = compute_edge_labels(rag, gtImage, n_threads = used_threads)
    z_edges = feats.compute_z_edge_mask(rag, watershed)
    # save train edge data
    # f = h5py.File('edge_gt.h5', 'w')
    # f.create_dataset('data', data=edge_labels)
    # f.close()
    # f = h5py.File('uv_ids.h5', 'w')
    # f.create_dataset('data', data=rag.uvIds())
    # f.close()
    # f = h5py.File('edge_indications_0.h5', 'w')
    # f.create_dataset('data', data=z_edges)
    # f.close()
    rf_xy, rf_z = learn_random_forests_for_xyz_edges(features, edge_labels, z_edges, edge_mask=None, n_threads = used_threads)
    import joblib
    # save model
    joblib.dump(rf_xy, save_path + '/' + 'rf_xy.pkl')
    joblib.dump(rf_z, save_path + '/' + 'rf_z.pkl')
    # # load model
    # rfc2 = joblib.load('saved_model/rfc.pkl')
    # # predict
    #features_test, rag_test = compute_feature(watershed, raw_train, boundaries_train)
    #z_edges = feats.compute_z_edge_mask(rag_test, watershed)
    edge_probs = predict_edge_random_forests_for_xyz_edges(rf_xy, rf_z, features, z_edges)
    return edge_probs, rag

def test_3D(raw_test, boundaries_test, watershed_test):
    # costs = feats.compute_boundary_mean_and_length(rag, boundaries)[:, 0]
    # 学习的过程
    costs, rag_test = learning_edge()
    edge_sizes = feats.compute_boundary_mean_and_length(rag_test, boundaries_test, n_threads = used_threads)[:, 1]
    z_edges = feats.compute_z_edge_mask(rag_test, watershed_test)
    xy_edges = np.logical_not(z_edges)
    edge_populations = [z_edges, xy_edges]
    costs = mc.transform_probabilities_to_costs(costs, beta=0.50, edge_sizes=edge_sizes,
                                     edge_populations=edge_populations)
    node_labels = mc.multicut_fusion_moves(rag_test, costs)

    # from elf.segmentation.clustering import mala_clustering, agglomerative_clustering
    # node_labels = mala_clustering(rag, costs, edge_sizes, 0.1)
    # map the results back to pixels to obtain the final segmentation
    segmentation = feats.project_node_labels_to_pixels(rag_test, node_labels)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw_test, name='raw')
        viewer.add_image(boundaries_test, name='boundaries')
        viewer.add_labels(watershed_test, name='watershed')
        viewer.add_labels(segmentation, name='multicut')
    # imageio.volwrite(r'\\192.168.3.2\data01\users\hongb\SCN_fine result\wafer14-4\segmentation_3D.tif', segmentation.astype('uint16'))
    # imageio.volwrite(r'\\192.168.3.2\data01\users\hongb\SCN_fine result\wafer14-4\boundaries.tif',
    #                  (boundaries_test*255).astype('uint8'))

def read_32_tif(name, whether_affinity):
    if whether_affinity == True:
        raw = imageio.volread(name)
    else:
        raw = imageio.imread(name)
    return raw

def read_32_tif_seqqence(image_dir, whether_affinity =  False, whether_with_= False):
    test_data_path = os.path.join(image_dir)
    images = os.listdir(test_data_path)
    if whether_with_ == True:
        images.sort(key=lambda x: int(x.split('_')[0]))  ##sort
    else:
        images.sort(key=lambda x: int(x.split('.')[0]))  ##sort
    total = int(len(images))  # total应该是整数才对

    image_name = image_dir + '/' + images[0]  # image_type
    image_temp = read_32_tif(image_name, whether_affinity)
    size_y = image_temp.shape[0]
    size_x = image_temp.shape[1]
    img = np.zeros((total, size_y, size_x), dtype=image_temp.dtype)
    del image_temp

    for i, filename in enumerate(images):
        image_name = image_dir + '/' + str(filename)  # image_type
        image_temp = read_32_tif(image_name, whether_affinity)
        img[i] = image_temp

    print('reading over')

    return img

def train_multicut(row_path, pre_path, gt_path, save_path):
    # load data
    raw_train = read_32_tif_seqqence(row_path)
    boundaries_train = (read_32_tif_seqqence(pre_path) / 255).astype(np.float32)  # 这里已经进行了1-，不需要读入反色图像
    gtImage = read_32_tif_seqqence(gt_path)

    watershed = ws.stacked_watershed(boundaries_train, threshold=.3, sigma_seeds=6., sigma_weights=5, min_size=300)[0]

    # show watershed
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw_train, name='raw')
        viewer.add_image(boundaries_train, name='boundaries')
        viewer.add_labels(watershed, name='watershed')

    # compute the region adjacency graph
    # rag = feats.compute_rag(watershed_test)
    edge_probs, rag_test = learning_edge(watershed, raw_train, boundaries_train, gtImage, save_path)

    edge_sizes = feats.compute_boundary_mean_and_length(rag_test, boundaries_train, n_threads = used_threads)[:, 1]
    z_edges = feats.compute_z_edge_mask(rag_test, watershed)
    xy_edges = np.logical_not(z_edges)
    edge_populations = [z_edges, xy_edges]
    costs = mc.transform_probabilities_to_costs(edge_probs, beta=.5, edge_sizes=edge_sizes,
                                                edge_populations=edge_populations)

    node_labels = mc.multicut_gaec(rag_test, costs)
    segmentation = project_node_labels_to_pixels(rag_test, node_labels)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw_train, name='raw')
        viewer.add_image(boundaries_train, name='boundaries')
        # viewer.add_labels(segmentation, name='multicut-lifted')
        viewer.add_labels(segmentation, name='multicut-local')

def load_data(row_path, pre_path, gt_path):
    # load data
    raw_train = read_32_tif_seqqence(row_path)
    boundaries_train = (1.-read_32_tif_seqqence(pre_path) / 255).astype(np.float32)  # 这里已经进行了1-，不需要读入反色图像
    gtImage = read_32_tif_seqqence(gt_path)

    watershed = ws.stacked_watershed(boundaries_train, threshold=.3, sigma_seeds=6., sigma_weights=5, min_size=300)[0]
    features, rag = compute_feature(watershed, raw_train, boundaries_train)
    edge_labels = compute_edge_labels(rag, gtImage, n_threads = used_threads)
    z_edges = feats.compute_z_edge_mask(rag, watershed)
    print(row_path,' loading over')
    return features, edge_labels, z_edges

def _mask_edges(rag, edge_costs):
    uv_ids = rag.uvIds()
    ignore_edges = (uv_ids == 0).any(axis=1)
    edge_costs[ignore_edges] = - np.abs(edge_costs).max()
    return edge_costs


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


def test_multicut(filelist, load_path, zhezhou_mask_path, counter_file_path):
    # load data       row_path_test, pre_path_test, xxxx, save_path, whether_affinity

        import time
    
        row_path_test = filelist[0]
        pre_path_test = filelist[1]
        affineDataPath = filelist[2]
        whether_affinity = False
        #save_path1 = filelist[3]
        save_path = filelist[3]
    
    
    
        #if not os.path.exists(save_path1):
        #    os.mkdir(save_path1)
    
        #if not os.path.exists(save_path2):
        #    os.mkdir(save_path2)
    
        t0 = time.time()
        t_start = t0
    
        raw_tests = read_chen_z_y_x(row_path_test, 'test')
        #raw_tests = read_32_tif_seqqence(row_path_test)
    
        ########################### BRIQU test for black area ###############################
    
        all = row_path_test.rsplit(".", 1)[0]
        row = int(all.split('_')[-2])
        line = int(all.split('_')[-1])
    
        stack_path = row_path_test.split('/layer')[0]
        min_z, max_z, min_y, max_y, min_x, max_x = get_affinity_information(stack_path)
        start_row = min_y
        end_row = max_y
        start_line = min_x
        end_line = max_x
        random_size_kuan = 1000
        random_size_chang = 1000
        resample = 3
        threshold_gradient = 1050
        threshold_var = 14
        # big_image = cv2.equalizeHist(raw_tests[0])
        big_image = raw_tests[0]
        count = 0
        
        #######check counter######
        stack_num = all.split('/')[-2][5:]
        counter_path = counter_file_path + str(int(stack_num)).zfill(4) + '.tif'
    
        counter = imageio.imread(counter_path)
    
        lunkuo_img = cv2.resize(counter, (160000, 160000), interpolation=cv2.INTER_NEAREST)
    
        row_zuobiao_start = (row - 1) * 8600 - (row - 1) * 600
        row_zuobiao_end = row * 8600 - (row - 1) * 600
        col_zuobiao_start = (line - 1) * 8600 - (line - 1) * 600
        col_zuobiao_end = line * 8600 - (line - 1) * 600
    
        counter_mask = lunkuo_img[row_zuobiao_start:row_zuobiao_end, col_zuobiao_start:col_zuobiao_end]
        error_area = np.sum(counter_mask) / 255
        stack_name = 'stack' + str(stack_num)
        print(stack_name + ' row: ' + str(row) + ' line: ' + str(
            line) + ' loss_area: ' + str(error_area))

        ##########################
        if error_area < 3000000:
            print(row_path_test + ' is < 3000000')
          
            for x_time in range(1, 3):
                for y_time in range(1, 3):
                    for time_x in range(resample):
                        image_part = get_location(big_image, x_time, y_time, row, start_row, end_row, line, start_line,
                                                  end_line, random_size_kuan, random_size_chang)
                        # plt.imshow(image_part, cmap='gray')
                        gradient, var = calcul_gradient_var(image_part)
        
                        if (whether_white_area(gradient, var, threshold_gradient, threshold_var) == False):
                            count = count + 1
            del image_part
        else:
            count = 0
    
        if count > 7:
            print(row_path_test, ' have too much black or white area')
            black_matrix = np.zeros_like(raw_tests)
            #if not os.path.exists(save_path1):
            #    os.makedirs(save_path1)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            name = save_path.split('/')[-1]
    
            with h5py.File(save_path + '/' + name + '.h5', 'w') as f:
                f.create_dataset('data', data=black_matrix, compression='gzip')
                f.create_dataset('shape', data=(black_matrix.shape[0], black_matrix.shape[1], black_matrix.shape[2]),
                                 compression='gzip')
    
            print('end computing ', pre_path_test)
            print('total time is ', time.time() - t_start)
        else:
        #######################################################################################
    
            test_pred_path = os.path.join(pre_path_test)
            pred_image_list = os.listdir(test_pred_path)
            pred_image_list.sort(key=lambda x: int(x.split('.')[0].split('_')[0]))
    
            total = len(pred_image_list)  # total应该是整数才对
    
            boundaries_tests = []
            for i in tqdm(range(total)):
    
                pre_name = pred_image_list[i]
    
                if whether_affinity == True:
                     aff = imageio.volread(pre_path_test + '/' + pre_name)
                     boundaries_test = 1 / 9 * aff[0] + 4 / 9 * aff[1] + 4 / 9 * aff[2]
                else:
                    boundaries_test = imageio.imread(pre_path_test + '/' + pre_name)
    
                ###
                boundaries_tests.append((1 - boundaries_test / 255).astype('float32'))  # 这里已经进行了1-，不需要读入反色图像
    
            boundaries_tests = np.stack(boundaries_tests, axis=0)
    
            pad_size = 1000 ############################para
            affine_scale = 10 ##########################para
    
            ##########zhezhou_mask###########
    
            # mask_path = '/home/liujz/3dian33_liujz/zebrafish_mask/zbf_1st_mask_noshift'
            # zhezhou_mask_path = '/home/liujz/3dian33_liujz/zebrafish_mask/zbf-2nd-test/stack186_maskAndData/zbf_186_mask'
    
            all = row_path_test.rsplit(".", 1)[0]
            stack_num = all.split('/')[-2][5:]
            row = int(all.split('_')[-2])
            line = int(all.split('_')[-1])
            small_step = np.uint8(8000 / 40);
            small_edge = np.uint8(600 / 40);
            # width = mask.shape[1];
            # height = mask.shape[0];
            mask_name_path = zhezhou_mask_path + '/' + 'zebrafish-stack' + stack_num + '/' + ''
    
            if not os.path.exists(mask_name_path):
                zhezhou_mask = np.zeros_like(boundaries_tests)
            else:
                mask_name_list = os.listdir(mask_name_path)
                mask_name_list.sort(key=lambda x: int(x.split('.')[0][5:]))
                zhezhou_mask = np.zeros_like(boundaries_tests)
                for mask_i in mask_name_list:
                    mask_temp = imageio.imread(mask_name_path + '/' + mask_i)
                    number = int(mask_i.split('.')[-2][5:])
    
                    block = mask_temp[(row - 1) * small_step:(row) * small_step + small_edge,
                            (line - 1) * small_step: (line) * small_step + small_edge]
    
                    ret, block = cv2.threshold(src=block,  # 要二值化的图片
                                               thresh=0,  # 全局阈值
                                               maxval=255,  # 大于全局阈值后设定的值
                                               type=cv2.THRESH_BINARY)
    
                    block = cv2.resize(block, (8600, 8600),
                                       interpolation=cv2.INTER_NEAREST)  # fx=40, fy=40, interpolation=cv2.INTER_LINEAR
                    # cv2.imwrite(
                    #     '/home/liujz/3dian33_liujz/zebrafish_mask/test_noshift/stack523/' + 'layer23_' + str(i + 1) + '_' + str(
                    #         j + 1) + '.jpg', block)
    
                    zhezhou_mask[number - 1] = block
            zhezhou_mask = np.uint8(zhezhou_mask)
            ##############################
    
            #########counter#########
    
            # counter_file_path = '/home/liujz/3dian33_data2/share/zebrafish_mask/brain_mask/DILATE_cbh_downsample_shift_brainmask/'
            counter_path = counter_file_path + str(int(stack_num) + 4).zfill(4) + '.tif'
    
            counter = imageio.imread(counter_path)
    
            # rows, cols = counter.shape
    
            # new_counter = affine_trans(counter, matrix2, 1, Whether_invert = False)
            # new_counter = cv2.warpAffine(counter, matrix3, (cols, rows))
            # cv2.imwrite('/home/liujz/3dian4/shenjingsuo_data/2022_data/count_aff_test/new_counter.jpg', new_counter)
    
            lunkuo_img = cv2.resize(counter, (160000, 160000), interpolation=cv2.INTER_NEAREST)
    
            row_zuobiao_start = (row - 1) * 8600 - (row - 1) * 600
            row_zuobiao_end = row * 8600 - (row - 1) * 600
            col_zuobiao_start = (line - 1) * 8600 - (line - 1) * 600
            col_zuobiao_end = line * 8600 - (line - 1) * 600
    
            counter_mask = lunkuo_img[row_zuobiao_start:row_zuobiao_end, col_zuobiao_start:col_zuobiao_end]
    
            counter_masks = np.repeat(counter_mask[np.newaxis, :, :], total, axis=0)
    
            counter_masks = 255 - counter_masks
    
            #mask = counter_masks + zhezhou_mask
            temp1 = np.uint8(zhezhou_mask/255)
            temp2 = np.uint8(counter_masks/255)
            mask = temp1 + temp2
            mask = mask > 0
            mask = mask * 255
            #can not add!!!
            mask = np.uint8(mask)
    
            del temp1,temp2
    
            ##############################
    
    
            boundaries_tests = np.pad(boundaries_tests, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'symmetric')  # constant
            boundaries_tests = affine_trans(boundaries_tests, row_path_test, affineDataPath, affine_scale, Whether_invert=False)
    
            raw_tests = np.pad(raw_tests, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'symmetric')  # constant
            raw_tests = affine_trans(raw_tests, row_path_test, affineDataPath, affine_scale, Whether_invert=False)
    
            mask = np.pad(mask, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'symmetric')  # constant
            mask = affine_trans(mask, row_path_test, affineDataPath, affine_scale, Whether_invert=False)
    
            #########################################
    
            #raw_backgrounds = raw_tests == 0
           # boundaries_tests = boundaries_tests + raw_backgrounds
            mask_back = mask == 255
    
            boundaries_tests = np.clip(boundaries_tests, 0, 1)
    
    
            name = pre_path_test.split('/')[-2]
            print('Reading ', name , ' takes time:', time.time() - t0)
    
            # watershed
            t0 = time.time()
            #watershed_test = ws.stacked_watershed(boundaries_tests, threshold=.15,
            #                                      sigma_seeds=3., sigma_weights=2,
            #                                      min_size=50, mask=~raw_backgrounds)[0]
    
            watershed_test = ws.stacked_watershed(boundaries_tests, threshold=.15,
                                                  sigma_seeds=3., sigma_weights=2,
                                                  min_size=50, mask=~mask_back)[0]
    
            print('watershed takes time:', time.time() - t0)
    
    
            # load model
            rf_xy = joblib.load(load_path + '/' + 'rf_xy.pkl')
            rf_z = joblib.load(load_path + '/' + 'rf_z.pkl')
    
            del test_pred_path, pred_image_list, zhezhou_mask
            del counter, lunkuo_img,  counter_mask,  counter_masks, mask
    
            # predict
            t0 = time.time()
            features_test, rag_test = compute_feature(watershed_test, raw_tests,
                                                      boundaries_tests)
            print('compute feature takes time:', time.time() - t0)
    
            t0 = time.time()
            z_edges = feats.compute_z_edge_mask(rag_test, watershed_test)
            print('compute_z_edge_mask takes time:', time.time() - t0)
    
            t0 = time.time()
            try:
                edge_probs = predict_edge_random_forests_for_xyz_edges(rf_xy, rf_z, features_test, z_edges)
            except:
            
                print(row_path_test, ' random_forests error')
                black_matrix = np.zeros_like(raw_tests)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                name = save_path.split('/')[-1]
    
                with h5py.File(save_path + '/' + name + '.h5', 'w') as f:
                    f.create_dataset('data', data=black_matrix, compression='gzip')
                    f.create_dataset('shape', data=(black_matrix.shape[0], black_matrix.shape[1], black_matrix.shape[2]),
                                     compression='gzip')
                
                            
                f = open('/opt/data/Nas402/zbrfish/liujz_code/' + '/' + 'error_2023.txt', mode='a')
                f.write('random_forests error:'+ '\t' + pre_path_test + '\n')  # write 写入
                f.close()  # 关闭文件
                
    
                print('end computing ', pre_path_test)
                print('total time is ', time.time() - t_start)
                return 0
              
            del raw_tests
            
            print('predict_edge_random_forests_for_xyz_edges takes time:', time.time() - t0)
    
            edge_sizes = feats.compute_boundary_mean_and_length(rag_test, boundaries_tests, n_threads = used_threads)[:, 1]
            z_edges = feats.compute_z_edge_mask(rag_test, watershed_test)
            xy_edges = np.logical_not(z_edges)
            edge_populations = [z_edges, xy_edges]
            costs = mc.transform_probabilities_to_costs(edge_probs, beta=0.55, edge_sizes=edge_sizes,
                                                        edge_populations=edge_populations)
            costs = _mask_edges(rag_test, costs)
            # np.save(r'costs_'+str(zaxis), costs)
    
            del edge_probs, rf_xy, rf_z
    
            # multicut
            t0 = time.time()
            #node_labels = mc.multicut_kernighan_lin(rag_test, costs)           #kl slow and hard
            node_labels = mc.multicut_gaec(rag_test, costs)                  #gaec quick
            # node_labels = mc.blockwise_multicut(rag_test, costs, watershed_test, internal_solver='fusion-moves',
            #                                      block_shape=[20, 2048, 2048], n_threads=32, n_levels=2, halo=[5, 500, 500])
            print('GAEC solve multicut takes time:', time.time() - t0)
    
            segmentation1 = feats.project_node_labels_to_pixels(rag_test, node_labels)
    
    
            ## post-processing
            from elf.segmentation.postprocess import graph_size_filter
            from elf.parallel.relabel import relabel_consecutive
            from elf.parallel.unique import unique
    
            t0 = time.time()
    
            foreground = (1. - boundaries_tests) > 0.3
            struct = np.ones((1, 5, 5))
            foreground = opening(foreground, struct)
            foreground = remove_small_objects(foreground.astype('bool'), min_size=500, connectivity=1, in_place=False)
            foreground = binary_fill_holes(foreground.astype('bool'))
            segmentation2 = segmentation1 * foreground.astype('uint32')  #############
    
    
            #### relabel
            relabel_consecutive(segmentation2.astype('uint32'), block_shape=[5, 2048, 2048], out=segmentation2)
    
            #### remove small area label
            seg_sizes = unique(segmentation2, return_counts=True, block_shape=[5, 2048, 2048])[1]
            label_s2 = unique(segmentation2, return_counts=True, block_shape=[5, 2048, 2048])[0]
            list_remove = np.where(seg_sizes < 1000)
    
            rag_seg2 = feats.compute_rag(segmentation2)
            node_labels_seg2 = label_s2
            discard_labels = label_s2[list_remove[0]]
            node_labels_seg2[discard_labels] = 0
            #edge_weigths = compute_boundary_mean_and_length(rag_seg1, boundaries_tests, n_threads = used_threads)[:, 0]
            #node_labels2 = graph_size_filter(rag_seg1, edge_weigths, seg_sizes, 2000)
    
            segmentation2 = feats.project_node_labels_to_pixels(rag_seg2, node_labels_seg2)
    
            print('post-processing takes time:', time.time() - t0)
            
            del segmentation1, boundaries_tests, rag_test, features_test
            del seg_sizes, label_s2, list_remove, rag_seg2, node_labels_seg2, discard_labels
            del foreground, watershed_test, z_edges, xy_edges, edge_populations, costs
    
            ###################reverse transform##################
            # seg1_reverse = np.zeros_like(segmentation1)
            # seg2_reverse = np.zeros_like(segmentation2)
            # for i in range(raw_tests.shape[0]):
            #     im1 = segmentation1[i].T
            #     im2 = segmentation2[i].T
            #     order = 0
            #     if i == 0:
            #         seg1_reverse[i] = im1.T
            #         seg2_reverse[i] = im2.T
            #     else:
            #         matrix = compute_matrix(matrix_cell, i)
            #         matrix = np.linalg.inv(matrix)
            #         seg1_reverse[i] = affine_transform(im1, matrix, order=order).T
            #         seg2_reverse[i] = affine_transform(im2, matrix, order=order).T
            # #######################################################
    
            #seg1_reverse = affine_trans(segmentation1, row_path_test, affineDataPath, affine_scale, Whether_invert=True)   #remove  segmentation1 information
            seg2_reverse = affine_trans(segmentation2, row_path_test, affineDataPath, affine_scale, Whether_invert=True)
    
            ##### crop reverse
            # out_reverse
            #seg1_result = seg1_reverse[:, pad_size:-pad_size, pad_size:-pad_size]    #remove  segmentation1 information
            seg2_result = seg2_reverse[:, pad_size:-pad_size, pad_size:-pad_size]
            # raw = raw[:, 500:-500, 500:-500]
            del segmentation2, seg2_reverse
    
            #####################################
            # #  save segmentation
            #if not os.path.exists(save_path1):
            #    os.makedirs(save_path1)
    
            if not os.path.exists(save_path):
                os.makedirs(save_path)
    
            #start_layer = row_path_test.split('/')[-1].split('_')[1].split('-')[0]
    
            #for i in range(segmentation1.shape[0]):                                                                                 #remove  segmentation1 information
            #    imageio.imwrite(save_path1 + '/' + str(i + int(start_layer)).zfill(4) + '.tif', seg1_result[i].astype('uint32'))    #remove  segmentation1 information
    
            #for i in range(segmentation2.shape[0]):
            #    imageio.imwrite(save_path2 + '/' + str(i + int(start_layer)).zfill(4) + '.tif', seg2_result[i].astype('uint32'))
    
            name = save_path.split('/')[-1]
    
            with h5py.File(save_path + '/' + name + '.h5', 'w') as f:
                 f.create_dataset('data', data=seg2_result, compression='gzip')
                 f.create_dataset('shape', data=(seg2_result.shape[0],seg2_result.shape[1], seg2_result.shape[2]), compression='gzip')
    
            print('end computer ', pre_path_test)
            print('total time is ', time.time() - t_start)



def run(whether_Train, train_path, pkl_save_path):

    if whether_Train == True:
        list = os.listdir(train_path)
        #list.pop()
        #list.pop()

        list_F = []
        list_edge = []
        list_z = []

        for i in list:
            row_path = train_path + i + '/' + 'raw_down_2'
            pre_path = train_path + i + '/' + 'train_probability_tz'
            gt_path = train_path + i + '/' + 'label_down_2'
            # train_multicut(row_path, pre_path, gt_path, pkl_save_path)
            list_f, edge_labels, z_edges = load_data(row_path, pre_path, gt_path)
            list_F.append(list_f)
            list_edge.append(edge_labels)
            list_z.append(z_edges)
        list_F = np.concatenate(list_F, axis=0)
        list_edge = np.concatenate(list_edge, axis=0)
        list_z = np.concatenate(list_z, axis=0)
        ###
        rf_xy, rf_z = learn_random_forests_for_xyz_edges(list_F, list_edge, list_z)
        import joblib
        # save model
        joblib.dump(rf_xy, pkl_save_path + '/' + 'rf_xy.pkl')
        joblib.dump(rf_z, pkl_save_path + '/' + 'rf_z.pkl')

if __name__ == '__main__':

    whether_Train = False

    if whether_Train == True:
        #train_path = '/home/liujz/bigstore/banmayu_12_10_4096/new_train_data_project/'
        #pkl_save_path = '/home/liujz/bigstore/banmayu_12_10_4096/savepkl/'

        train_path = '/home/liujz/bigstore/ninanjie/save/ninanjie_output_cutted/'
        pkl_save_path = '/home/liujz/bigstore/ninanjie/save/'

        run(whether_Train, train_path, pkl_save_path)

    whether_Test = True

    if whether_Test == True:



        outPutList_stack = [] 
        for stack in range(461, 462): #524, 527 #530       #436, 467   #248, 336     #248, 303
            stack_name = 'stack' + str(stack)
            img_path = '/opt/data/Nas402/zbrfish/zbfStackResults_first_enhance/'
            #pre_path = '/opt/data/RecData/zbfStackResults_first_probability/'
            pre_path = '/opt/data/RecData/zbfStackResults_first_probability/'
            matrix_path = '/opt/data/RecData/zbfStackResults_first_matrix/'
            #save_path1 = '/opt/data/3_33/zebrafish/zbfStackResults_multicut/'
            save_path = '/opt/data/Nas402/zbrfish/zbfStackResults_first_multicut/'
            outPutList = get_processing_information(stack_name, img_path, pre_path, matrix_path, save_path)
            outPutList_stack.extend(outPutList)
        #pkl_load_path = '/home/liujz/bigstore/banmayu_12_10_4096/multicut_pkl/'
        pkl_load_path = '/opt/data/Nas402/zbrfish/multicut_para/savepkl_20220718/'
        zhezhou_mask_path = '/opt/data/Nas402/zbrfish/zbf_mask/zhezhou/zbf_1st_mask_noshift/'
        counter_file_path = '/opt/data/Nas402/zbrfish/zbf_mask/counter/'

        print(len(outPutList_stack))
        for x in outPutList_stack:
            print(x[0])
            test_multicut(x, pkl_load_path, zhezhou_mask_path, counter_file_path)


        

