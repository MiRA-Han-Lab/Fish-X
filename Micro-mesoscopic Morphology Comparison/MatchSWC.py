# this file is for LM and EM morphological comparision, getting each LM and EM swc from csv lists that you got in step 1 and 2, and compute the morphology similarity(distance) of each pair and save as TOP10 csv for each EM neuron
import os
import numpy as np
import csv
from scipy.spatial import KDTree
import time

'''
demo
EM=np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
LM=np.array([[1,2,4],[1,2,4],[1,2,3],[1,2,3],[1,2,3]])
delta=a[0]-c[0]
LM_offset=LM+delta
print(LM_offset)

'''
def read_swc(file_path):
    swc_data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            if len(line)<2:
                continue
            cols = line.split()

            x, y, z = float(cols[2]), float(cols[3]), float(cols[4])
            swc_data.append([ x, y, z])
    return np.array(swc_data)

def calculate_sd(true_positions, guang):
    tree1 = KDTree(guang)
    distances1, _ = tree1.query(true_positions)
    sd = np.mean(distances1)
    return sd

#read EM csv file
csv_reader1 = csv.reader(open("EM495_proofed2_r1.csv")) #change to totalEM csv name

# Read the target SWC file
LM_folder_path='/home/testuser1/User_folder/ZJL/SyN_swc_total/'
EM_folder_path='/home/testuser1/User_folder/ZJL/dictionary/EM_FM/zfishall_bind_neuron_proof_2_swc_result1/'

index=1
for row in csv_reader1:
  EM_swc_name=str(row[0])
  EM_path=os.path.join(EM_folder_path,EM_swc_name)
  EM = read_swc(EM_path)

  #Acquisition of soma position
  Em_soma=EM[0]

  matches = [] # Initialize a list to store the matching results
  
  LM_path='/home/testuser2/User_folder/ZJL/EM_LM/LMneighbors_list/'+str(EM_swc_name)+'_LM_neighbours.csv' #read LM_neighber csv file
  csv_reader2 = csv.reader(open(LM_path)) 
  
  for row2 in csv_reader2:
     swc_file = str(row2[0])
     if 'vglut2a' in swc_file or 'vglut2b' in swc_file:
        swc_path = os.path.join(LM_folder_path, swc_file)
        LM = read_swc(swc_path)

        LM_soma=LM[0]
        delta=Em_soma-LM_soma
        LM_offset=LM+delta

        LM_type = 'E'
        # Calculate the similarity with the target SWC file
        distance = calculate_sd(EM, LM_offset)
        # Add the result to the matching list
        matches.append((swc_file, distance, LM_type))
     if 'glyt2' in swc_file or 'gad1b' in swc_file:
        swc_path = os.path.join(LM_folder_path, swc_file)
        LM = read_swc(swc_path)
        LM_soma = LM[0]
        delta = Em_soma - LM_soma
        LM_offset = LM + delta
        LM_type = 'I'
        # Calculate the similarity with the target SWC file
        distance = calculate_sd(EM, LM_offset)
        # Add the result to the matching list
        matches.append((swc_file, distance, LM_type))
  
  # Sort the matching results based on similarity
  matches.sort(key=lambda x: x[1])
 
  # output top10 neighbour LM swc and write into csv
  with open('/home/testuser2/User_folder/ZJL/EM_LM/LMneighbors_top10list/'+str(EM_swc_name)+'_LM_neighbours_top.csv','w',newline='') as csvfile:
  	writer=csv.writer(csvfile)
  	if len(matches)<10:
  		t=len(matches)
  	else:
  		t=10
  		
  	for match in matches[:t]:
  		writer.writerow([match[0],match[1],match[2]])
  index +=1



