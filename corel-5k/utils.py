# coding: utf-8
import os
import math
import random
import numpy as np
import pickle

from skimage import io,transform
import matplotlib.pyplot as plt
import pickle

import warnings
warnings.filterwarnings("ignore")

def load_data(ROOT="./Datasets/corel_5k/"):
    """
    Args: 
        ROOT        : the root location of the corel 5k dataset.
    Outputs:
        files       : the path of all of the images included
        train_pairs : the path of the train  images and their labels' index list
        val_pairs   : the path of the val    images and their labels' index list
        test_pairs  : the path of the test   images and their labels' index list
    """
    IMAROOT = ROOT+'images/'
    LABROOT = ROOT+'labels/'
    dirs = [IMAROOT+i+"/" for i in next(os.walk(IMAROOT))[1]]
    files = []
    top100labels = pickle.load(open('./source/data/top100labels.pkl','rb'))
    [files.extend([i+j for j in next(os.walk(i))[2] if "jpeg" in j]) for i in dirs]

    with open(LABROOT+"training_label") as f:
        train_labels = f.readlines()
    train_labels = [i.split(" ")[:] for i in train_labels]
    train_labels = [[int(j) for j in i if j != '' and j != '\n']for i in train_labels]
    random.shuffle(train_labels)
    train_label = train_labels[:]
    val_label = train_labels[4000:]

    train_label_dict = {}
    for i in train_label:
        temp = [lab for lab in i[1:] if lab in top100labels]
        if temp!=[]:
            train_label_dict[str(i[0])+".jpeg"] = temp
        
    val_label_dict = {}
    for i in val_label:
        temp = [lab for lab in i[1:] if lab in top100labels]
        if temp!=[]:
            val_label_dict[str(i[0])+".jpeg"] = temp
        
    with open(LABROOT+"test_label") as f:
        test_labels = f.readlines()
    test_labels = [i.split(" ")[:] for i in test_labels]
    test_labels = [[int(j) for j in i if j != '' and j != '\n']for i in test_labels]
    test_label_dict = {}
    for i in test_labels:
        temp = [lab for lab in i[1:] if lab in top100labels]
        if temp!=[]:
            test_label_dict[str(i[0])+".jpeg"] = temp

    train_pairs  = []
    val_pairs    = []
    test_pairs   = []
    for i in files:
        img_name = i.split("/")[-1]
        if img_name in val_label_dict.keys():
            val_pairs.append((i, val_label_dict[img_name]))
        elif img_name in test_label_dict.keys():
            test_pairs.append((i, test_label_dict[img_name]))
        elif img_name in train_label_dict.keys():
            train_pairs.append((i, train_label_dict[img_name]))

    print(len(train_pairs),len(test_pairs),train_pairs[1][1])
    return files, train_pairs, val_pairs, test_pairs

def load_class(ROOT="./Datasets/corel_5k/labels/"):
    """
    Args: 
        ROOT        : the root location of the corel 5k dataset's labels.
    Outputs:
        class_names : the list of classes' names 
    """    
    with open(ROOT+"words") as f:
        class_names = f.readlines()    
    class_names = [i.strip('\n') for i in class_names]
    return class_names

def get_weight(train_pairs,test_pairs,val_pairs):
    a = {}
    labels = []
    [labels.extend(i[1]) for i in train_pairs]
    [labels.extend(i[1]) for i in test_pairs]
    [labels.extend(i[1]) for i in val_pairs]
    for i in labels:
        if i in a.keys():
            a[i] += 1
        else:
            a[i] = 1
    for i in a.keys():
        a[i] = 1/a[i]
    weights = list(a.values())
    return weights

def get_mean(ROOT="./Datasets/corel_5k/images/"):
    """
    PIL.Image Ver:
        R_mean:98.2839  G_mean:102.1193 B_mean:88.5399
        R_std:64.3600   G_std:61.4550   B_std:63.8599
    torch.FloatTensor Ver:
        R_mean:0.3854   G_mean:0.4005   B_mean:0.3472
        R_std:0.2524    G_std:0.2410    B_std:0.2504
    """
    try:
        image_statistics = pickle.load(open('./source/image_statistics.pkl', "rb"))
    except:
        dirs = [ROOT+i+"/" for i in next(os.walk(ROOT))[1]]
        files = []
        [files.extend([i+j for j in next(os.walk(i))[2] if "jpeg" in j]) for i in dirs]
        r=[];g=[];b=[]
        for file in files:
            tmp_img = io.imread(file)
            r.extend(tmp_img[:,:,0].reshape(-1))
            g.extend(tmp_img[:,:,1].reshape(-1))
            b.extend(tmp_img[:,:,2].reshape(-1))
        r = np.array(r);g = np.array(g); b = np.array(b)
        r_mean = r.mean();g_mean = g.mean();b_mean = b.mean()
        r_std  = r.std() ;g_std  = g.std() ;b_std  = b.std()
        image_statistics = {'r_mean':r_mean,'g_mean':g_mean,'b_mean':b_mean,\
        'r_std':r_std,'g_std':g_std,'b_std':b_std}
        with open('./source/image_statistics.pkl','wb') as f:
            pickle.dump(image_statistics,f)
    return image_statistics

def folder_init():
    """
    Initialize folders required
    """
    if os.path.exists('source')            ==False:os.mkdir('source')
    if os.path.exists('source/log')        ==False:os.mkdir('source/log')
    if os.path.exists('source/trained_net')==False:os.mkdir('source/trained_net')
    if os.path.exists('source/val_results')==False:os.mkdir('source/val_results')
    

