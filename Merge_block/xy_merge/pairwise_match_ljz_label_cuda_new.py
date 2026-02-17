import sys
from collections import defaultdict
import numpy as np

import fast64counter
import imageio
import time
import matplotlib.pyplot as plt
import networkx as nx

import torch
import time
import fastremap

def update_label(label, height, hang, line, list_old, list_new, direction, cuda_num = 0, width=15):


    if isinstance(width, int):

        num_x = label.shape[2]
        num_y = label.shape[1]
        num_z = label.shape[0]
        bit_range = label.shape[3]

        if direction == 2:

            x_left_width = width if (line >= width) else line
            x_right_width = width if ((num_x - line) >= width) else (num_x - line)
            y_up_width = width if (hang >= width) else hang
            y_down_width = width if ((num_y - hang) >= width) else (num_y - hang)
            z_up_width = width if (height >= width) else height

            cuda_label_num_up = (x_left_width + x_right_width) * (y_up_width + y_down_width) * (z_up_width)
            cuda_label_num_down_up = (x_left_width + x_right_width) * y_up_width
            cuda_label_num_down_left = x_left_width + 1

            cuda_label_num = cuda_label_num_up + cuda_label_num_down_up + cuda_label_num_down_left

            label_selection = np.zeros((cuda_label_num, bit_range), dtype=np.int32)

            l = 0
            for z in range(height - z_up_width, height):
                for y in range(hang - y_up_width, hang + y_down_width):
                    for x in range(line - x_left_width, line + x_right_width):
                        #for l in range(0, cuda_label_num_up):
                        label_selection[l] = label[z, y, x]
                        l = l + 1
            if l!=(cuda_label_num_up):
                print('l != cuda_label_num_up :error-line52')

            l = cuda_label_num_up
            for y in range(hang - y_up_width, hang):
                for x in range(line - x_left_width, line + x_right_width):
                    #for l in range(cuda_label_num_up, cuda_label_num_up + cuda_label_num_down_up):
                    label_selection[l] = label[height, y, x]
                    l = l + 1
            if l!=(cuda_label_num_up + cuda_label_num_down_up):
                print('l != cuda_label_num_up + cuda_label_num_down_up :error-line61')

            l = cuda_label_num_up + cuda_label_num_down_up
            for x in range(line - x_left_width, line + 1):
                #for l in range(cuda_label_num_up + cuda_label_num_down_up, cuda_label_num):
                label_selection[l] = label[height, hang, x]
                l = l + 1
            if l!=(cuda_label_num):
                print('l != cuda_label_num :error-line69')

            label_selection_torch = torch.tensor(label_selection).cuda(cuda_num)

            for l in range(0, cuda_label_num):
                for i, j in zip(list_old, list_new):
                    whether = (label_selection_torch[l] == i)
                    label_selection_torch[l][whether] = j

            label_selection_cpu = label_selection_torch.cpu().numpy()

            l = 0
            for z in range(height - z_up_width, height):
                for y in range(hang - y_up_width, hang + y_down_width):
                    for x in range(line - x_left_width, line + x_right_width):
                        #for l in range(0, cuda_label_num_up):
                        label[z, y, x] = label_selection_cpu[l]
                        l = l + 1
            if l!=(cuda_label_num_up):
                print('l != cuda_label_num_up :error-line88')

            l = cuda_label_num_up
            for y in range(hang - y_up_width, hang):
                for x in range(line - x_left_width, line + x_right_width):
                    #for l in range(cuda_label_num_up, cuda_label_num_up + cuda_label_num_down_up):
                    label[height, y, x] = label_selection_cpu[l]
                    l = l + 1
            if l!=(cuda_label_num_up + cuda_label_num_down_up):
                print('l != cuda_label_num_up + cuda_label_num_down_up :error-line97')

            l = cuda_label_num_up + cuda_label_num_down_up
            for x in range(line - x_left_width, line + 1):
                #for l in range(cuda_label_num_up + cuda_label_num_down_up, cuda_label_num):
                label[height, hang, x] = label_selection_cpu[l]
                l = l + 1
            if l!=(cuda_label_num):
                print('l != cuda_label_num :error-line105')

        elif direction == 1:

            y_up_width = width if (hang >= width) else hang
            y_down_width = width if ((num_y - hang) >= width) else (num_y - hang)
            z_up_width = width if (height >= width) else height
            #z_down_width = width if (num_z - height >= width) else (num_z - height)

            cuda_label_num_up = z_up_width * (num_x * (y_up_width + y_down_width))
            cuda_label_num_down = (y_up_width + 1) * num_x
            cuda_label_num = cuda_label_num_up + cuda_label_num_down

            label_selection = np.zeros((cuda_label_num, bit_range), dtype=np.int32)

            l = 0
            for z in range(height - z_up_width, height):
                for y in range(hang - y_up_width, hang + y_down_width):
                    for x in range(0, num_x):
                        #for l in range(0, cuda_label_num_up):
                        label_selection[l] = label[z, y, x]
                        l = l + 1
            if l!=(cuda_label_num_up):
                print('l != cuda_label_num_up: error-line128')

            l = cuda_label_num_up
            for y in range(hang - y_up_width, hang + 1):
                for x in range(0, num_x):
                    #for l in range(cuda_label_num_up, cuda_label_num_up + cuda_label_num_down):
                    label_selection[l] = label[height, y, x]
                    l = l + 1
            if l!=(cuda_label_num_up + cuda_label_num_down):
                print('l != cuda_label_num_up + cuda_label_num_down :error-line137')

            label_selection_torch = torch.tensor(label_selection).cuda(cuda_num)

            for l in range(0, cuda_label_num):
                for i, j in zip(list_old, list_new):
                    whether = (label_selection_torch[l] == i)
                    label_selection_torch[l][whether] = j

            label_selection_cpu = label_selection_torch.cpu().numpy()

            ###
            l = 0
            for z in range(height - z_up_width, height):
                for y in range(hang - y_up_width, hang + y_down_width):
                    for x in range(0, num_x):
                        #for l in range(0, cuda_label_num_up):
                        label[z, y, x] = label_selection_cpu[l]
                        l = l + 1
            if l!=(cuda_label_num_up):
                print('l != cuda_label_num_up :error-line157')

            l = cuda_label_num_up
            for y in range(hang - y_up_width, hang + 1):
                for x in range(0, num_x):
                    #for l in range(cuda_label_num_up, cuda_label_num_up + cuda_label_num_down):
                    label[height, y, x] = label_selection_cpu[l]
                    l = l + 1
            if l!=(cuda_label_num_up + cuda_label_num_down):
                print('l != cuda_label_num_up + cuda_label_num_down :error-line166')

        elif direction == 0:

            z_up_width = width if (height >= width) else height

            cuda_label_num = (z_up_width + 1) * (num_x * num_y)

            if cuda_label_num < 64:

                label_selection = np.zeros((cuda_label_num, bit_range), dtype=np.int32)

                l = 0
                for z in range(height - z_up_width, height + 1):
                    for y in range(0, num_y):
                        for x in range(0, num_x):
                            #for l in range(0, cuda_label_num):
                            label_selection[l] = label[z, y, x]
                            l = l + 1
                if l != (cuda_label_num):
                    print('l != cuda_label_num :error-line186')


                label_selection_torch = torch.tensor(label_selection).cuda(cuda_num)

                for l in range(0, cuda_label_num):
                    for i, j in zip(list_old, list_new):
                        whether = (label_selection_torch[l] == i)
                        label_selection_torch[l][whether] = j

                label_selection_cpu = label_selection_torch.cpu().numpy()

                l = 0
                for z in range(height - z_up_width, height + 1):
                    for y in range(0, num_y):
                        for x in range(0, num_x):
                            #for l in range(0, cuda_label_num):
                            label[z, y, x] = label_selection_cpu[l]
                            l = l + 1
                if l != (cuda_label_num):
                    print('l !=cuda_label_num :error-line206')

            else:

                cuda_label_num_everylayer = num_x * num_y

                for z in range(height - z_up_width, height + 1):

                    label_selection = np.zeros((cuda_label_num_everylayer, bit_range), dtype=np.int32)

                    l = 0
                    for y in range(0, num_y):
                        for x in range(0, num_x):
                            #for l in range(0, cuda_label_num):
                            label_selection[l] = label[z, y, x]
                            l = l + 1
                    if l != (cuda_label_num_everylayer):
                        print('l !=cuda_label_num_everylayer :error-line223')

                    label_selection_torch = torch.tensor(label_selection).cuda(cuda_num)

                    for l in range(0, cuda_label_num_everylayer):
                        for i, j in zip(list_old, list_new):
                            whether = (label_selection_torch[l] == i)
                            label_selection_torch[l][whether] = j

                    label_selection_cpu = label_selection_torch.cpu().numpy()

                    l = 0
                    for y in range(0, num_y):
                        for x in range(0, num_x):
                            #for l in range(0, cuda_label_num):
                            label[z, y, x] = label_selection_cpu[l]
                            l = l + 1
                    if l != (cuda_label_num_everylayer):
                        print('l !=cuda_label_num_everylayer :error-line241')

    else:

        num_x = label.shape[2]
        num_y = label.shape[1]
        num_z = label.shape[0]
        bit_range = label.shape[3]
        label_torch = torch.tensor(label).cuda(cuda_num)

        if direction == 2:

            # for num_height in range(0, height):
            #     for num_hang in range(0, num_y):
            #         for num_line in range(0, num_x):
            #             for i, j in zip(list_old, list_new):
            #                 whether = (label_torch[num_height][num_hang][num_line] == i)
            #                 label_torch[num_height][num_hang][num_line][whether] = j
            #
            # for num_hang in range(0, hang):
            #     for num_line in range(0, num_x):
            #         for i, j in zip(list_old, list_new):
            #             whether = (label_torch[height][num_hang][num_line] == i)
            #             label_torch[height][num_hang][num_line][whether] = j

            for num_line in range(0, line + 1):
                for i, j in zip(list_old, list_new):
                    whether = (label_torch[height][hang][num_line] == i)
                    label_torch[height][hang][num_line][whether] = j

        elif direction == 1:
            # for num_height in range(0, height):
            #     for num_hang in range(0, num_y):
            #         for num_line in range(0, num_x):
            #             for i, j in zip(list_old, list_new):
            #                 whether = (label_torch[num_height][num_hang][num_line] == i)
            #                 label_torch[num_height][num_hang][num_line][whether] = j

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



def pair_match(block1_part, block2_part, height, hang, line, label, direction, halo_size, cuda_num=0, width = 15):

    direction = int(direction)
    halo_size = int(halo_size)

    ###############################
    # Note: direction indicates the relative position of the blocks (3, 2, 1 =>
    # adjacent in X, Y, Z).  Block1 is always closer to the 0,0,0 corner of the
    # volume.
    ###############################

    auto_join_pixels = 10000#20000 # Join anything above this many pixels overlap
    minoverlap_pixels = 2000#2000 # Consider joining all pairs over this many pixels overlap
    minoverlap_dual_ratio = 0.6#0.5#0.7 # If both overlaps are above this then join
    minoverlap_single_ratio = 0.8#0.8#0.9# If either overlap is above this then join

# Join 2 (more joining)
# auto_join_pixels = 10000; # Join anything above this many pixels overlap
# minoverlap_pixels = 1000; # Consider joining all pairs over this many pixels overlap
# minoverlap_dual_ratio = 0.5; # If both overlaps are above this then join
# minoverlap_single_ratio = 0.8; # If either overlap is above this then join

    #stacked = np.concatenate([block1, block2], axis=direction-1)############注意，这个地方我改了
    stacked = np.concatenate([block1_part, block2_part], axis=direction - 1)  ############注意，这个地方我改了

    #inverse, packed = np.unique(stacked, return_inverse=True)
    inverse, packed = fastremap.unique(stacked, return_inverse=True)


    #inverse, s, packed = np.unique(stacked, return_index=True, return_inverse=True)
    packed = packed.reshape(stacked.shape)

    if direction==3:###3
        packed_block1 = packed[:, :, :block1_part.shape[2]]
        packed_block2 = packed[:, :, block1_part.shape[2]:]
    elif direction==2:
        packed_block1 = packed[:, :block1_part.shape[1], :]
        packed_block2 = packed[:, block1_part.shape[1]:, :]
    elif direction==1:###1
        packed_block1 = packed[:block1_part.shape[0], :, :]
        packed_block2 = packed[block1_part.shape[0]:, :, :]


    # extract overlap

    ##lo_block1 = [0, 0, 0]
    ##hi_block1 = [None, None, None]
    ##lo_block2 = [0, 0, 0]
    ##hi_block2 = [None, None, None]

    # Adjust for Matlab HDF5 storage order
    #direction = 3 - direction
    direction = direction - 1

    # Adjust overlapping region boundaries for direction
    ##lo_block1[direction] = - 1 * halo_size
    ##hi_block2[direction] = 1 * halo_size

    ##block1_slice = tuple(slice(l, h) for l, h in zip(lo_block1, hi_block1))
    ##block2_slice = tuple(slice(l, h) for l, h in zip(lo_block2, hi_block2))
    packed_overlap1 = packed_block1
    packed_overlap2 = packed_block2
    print ("block1",  packed_overlap1.shape)
    print ("block2",  packed_overlap2.shape)

    counter = fast64counter.ValueCountInt64()
    counter.add_values_pair32(packed_overlap1.astype(np.int32).ravel(), packed_overlap2.astype(np.int32).ravel())
    overlap_labels1, overlap_labels2, overlap_areas = counter.get_counts_pair32()

    areacounter = fast64counter.ValueCountInt64()
    areacounter.add_values(packed_overlap1.ravel())
    areacounter.add_values(packed_overlap2.ravel())
    areas = dict(zip(*areacounter.get_counts()))

    to_merge = []
    to_steal = []
    #merge_dict = {}
    for l1, l2, overlap_area in zip(overlap_labels1, overlap_labels2, overlap_areas):

        #if l1 == 0 or l2 == 0:
        #    continue
        # find by jiangliuyun

        if inverse[l1] == 0 or inverse[l2] == 0:
            continue


        #if l1 == 5204 or l2 == 5204:
        #    aa = 1


        # if ((overlap_area > auto_join_pixels) or
        #      ((overlap_area > minoverlap_pixels) and
        #       ((overlap_area > minoverlap_single_ratio * areas[l1]) or
        #        (overlap_area > minoverlap_single_ratio * areas[l2]) or
        #        ((overlap_area > minoverlap_dual_ratio * areas[l1]) and
        #         (overlap_area > minoverlap_dual_ratio * areas[l2]))))):

        if ((overlap_area > minoverlap_single_ratio * areas[l1]) or
            (overlap_area > minoverlap_single_ratio * areas[l2]) or
            ((overlap_area > minoverlap_dual_ratio * areas[l1]) and
            (overlap_area > minoverlap_dual_ratio * areas[l2]))):

            #if inverse[l2] in merge_dict:
            #    if overlap_area < merge_dict[inverse[l2]][1]:
            #        continue

            if inverse[l1] != inverse[l2]:
                # print "Merging segments {0} and {1}.".format(inverse[l1], inverse[l2])
                to_merge.append((inverse[l1], inverse[l2]))
             #   merge_dict[inverse[l2]]=(inverse[l1],overlap_area)
        else:
            # print "Stealing segments {0} and {1}.".format(inverse[l1], inverse[l2])
            to_steal.append((overlap_area, l1, l2))

 


    merge_map = list((sorted(s, reverse=True)) for s in to_merge)
    merge_map = [value for index, value in
                 sorted(enumerate(merge_map), key=lambda merge_map: merge_map[1], reverse=True)]
    # merge_left = [x[0] for x in merge_map]
    # merge_right = [x[1] for x in merge_map]
    # merge_list = np.unique(merge_left + merge_right)

    ######test######
    #aa= []
    #for i in merge_map:
    #    if i[0] == 5204 or i[1] == 5204:
    #        aa.append(i)
    ################

    ##############graph################
    list_old = []
    list_new = []
    G = nx.Graph()
    G.add_edges_from(merge_map)
    # 打印连通子图
    for c in nx.connected_components(G):
        # 得到不连通的子集
        nodeSet = G.subgraph(c).nodes()

        target = min(nodeSet)
        for label_merge in nodeSet:
            #if label_merge == 5198:
            #    aaa = 1
            #if label_merge == 15:
            #    bbb = 1

            if label_merge != target:
                list_old.append(label_merge)
                list_new.append(target)
    ##############graph################

    


    #list_old

    start = time.process_time()
    label = update_label(label, height, hang, line, list_old, list_new, direction, cuda_num, width) ### GPU tensor speed up
    end = time.process_time()

    print(end-start)

    return label



