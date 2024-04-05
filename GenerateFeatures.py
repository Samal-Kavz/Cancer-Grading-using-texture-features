import cv2
import math
import numpy as np
import os
import pandas as pd

def calc_texture_features(glcm,feats):
    contrast = 0.0
    entropy = 0.0
    asm = 0.0
    idm = 0.0
    dis = 0.0
    mean_i = 0.0
    mean_j = 0.0
    gray_levels = 16
    for i in range(gray_levels):
        for j in range(gray_levels):
            contrast += (i-j)*(i-j)*glcm[i][j]
            asm += glcm[i][j]*glcm[i][j]
            idm += glcm[i][j]/(1+(i-j)*(i-j))
            dis += abs(i-j)*glcm[i][j]
            mean_i += i*glcm[i][j]
            mean_j += j*glcm[i][j]
            if (glcm[i][j]>0.0):
                entropy += glcm[i][j]*math.log(glcm[i][j])
            
    entropy = -entropy
    energy = asm**(1/2)

    var_i_2 = 0.0
    var_j_2 = 0.0
    for i in range(gray_levels):
        for j in range(gray_levels):
            var_i_2 += ((i-mean_i)**2)*glcm[i][j]
            var_j_2 += ((j-mean_j)**2)*glcm[i][j]

    var_i = var_i_2**(1/2)
    var_j = var_j_2**(1/2)

    correlation = 0.0
    shade = 0.0
    pro = 0.0
    for i in range(gray_levels):
        for j in range(gray_levels):
            correlation += (((i-mean_i)*(j-mean_j))/((var_i_2*var_j_2)**(1/2)))*glcm[i][j]
            shade += (i+j-mean_i-mean_j)**3*glcm[i][j]
            pro += (i+j-mean_i-mean_j)**4*glcm[i][j]
    
    feats.append(contrast)
    
    feats.append(entropy)
    
    feats.append(asm)
    
    feats.append(idm)
    
    feats.append(energy)
    
    feats.append(dis)
    
    feats.append(mean_i)
    
    feats.append(mean_j)
    
    feats.append(var_i)
    
    feats.append(var_j)
    
    feats.append(correlation)
    
    feats.append(shade)
    
    feats.append(pro)
 
def create_glcm_cimage(image,index):
    img_shape = image.shape
    #print(img_shape)
    img = cv2.resize(image,(img_shape[1]//2,img_shape[0]//2),interpolation=cv2.INTER_CUBIC)
    #srcdata = img.copy()
    gray_levels = 16
    glcm = [[0.0 for i in range(gray_levels)] for j in range(gray_levels)]
    height = img.shape[0]
    width = img.shape[1]
    img_2d = [[0 for i in range(width+1)] for j in range(height+1)]
    #print(len(img_2d))
    max_gray_level = 0
    dy=0
    dx=1
    #print(img)
    for i in range(height):
        for j in range(width):
            img_2d[i][j] = img[i][j][index]
            if(img[i][j][index]>max_gray_level):
                max_gray_level = img[i][j][index]
    max_gray_level += 1
    
    if(max_gray_level>gray_levels):
        for j in range(height):
            for i in range(width):
                img_2d[j][i] = img_2d[j][i]*gray_levels/max_gray_level

    for j in range(height-dy):
        for i in range(width-dx):
            rows = int(img_2d[j][i])
            cols = int(img_2d[j+dy][i+dx])
            glcm[rows][cols] += 1.0 #incrementing relevent location of glcm
        
    #print(glcm)
    #print(len(glcm))

    '''normalization of glcm'''
    for i in range(gray_levels):
        for j in range(gray_levels):
            glcm[i][j] /= float(height*width)
            
    return glcm

def create_glcm(img):
    try:
        img_shape = img.shape #dimensions of the image
    except:
        print("error while reading image")

    img = cv2.resize(img,(img_shape[1]//2,img_shape[0]//2),interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting rgb image to gray image

    srcdata = img_gray.copy()
    gray_levels = 16 #number of rows and columns glcm have
    glcm = [[0.0 for i in range(gray_levels)] for j in range(gray_levels)]

    (height,width) = img_gray.shape

    max_gray_level = 0
    '''finding maximum gray level of the gray image'''
    for i in range(height):
        for j in range(width):
            if(img_gray[i][j]>max_gray_level):
                max_gray_level = img_gray[i][j]
    max_gray_level += 1

    '''scaling gray level to get 16 gray levels'''
    if(max_gray_level>gray_levels):
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i]*gray_levels/max_gray_level
                
    dy = 0 #distance
    dx = 1 #angle
    for j in range(height-dy):
        for i in range(width-dx):
            rows = srcdata[j][i]
            cols = srcdata[j+dy][i+dx]
            glcm[rows][cols] += 1.0 #incrementing relevent location of glcm

    '''normalization of glcm'''
    for i in range(gray_levels):
        for j in range(gray_levels):
            glcm[i][j] /= float(height*width)
    
    return glcm

def create_features_set(image):
    names = []
    feats  = []
    
    names.append("contrast")
    names.append("entropy")
    names.append("asm")
    names.append("idm")
    names.append("energy")
    names.append("dissimilarity")
    names.append("mean_reference")
    names.append("mean_neighbor")
    names.append("var_refference")
    names.append("var_neighbor")
    names.append("correlation")
    names.append("shade")
    
    names.append("prominence")
    
    names.append("contrast_r")
    
    names.append("entropy_r")
    
    names.append("asm_r")
    
    names.append("idm_r")
    
    names.append("energy_r")
    
    names.append("dissimilarity_r")
    
    names.append("mean_reference_r")
    
    names.append("mean_neighbor_r")
    
    names.append("var_refference_r")
    
    names.append("var_neighbor_r")
    
    names.append("correlation_r")
    
    names.append("shade_r")
    
    names.append("prominence_r")
    
    names.append("contrast_g")
    
    names.append("entropy_g")
    
    names.append("asm_g")
    
    names.append("idm_g")
    
    names.append("energy_g")
    
    names.append("dissimilarity_g")
    
    names.append("mean_reference_g")
    
    names.append("mean_neighbor_g")
    
    names.append("var_refference_g")
    
    names.append("var_neighbor_g")
    
    names.append("correlation_g")
    
    names.append("shade_g")
    
    names.append("prominence_g")
    
    names.append("contrast_b")
    
    names.append("entropy_b")
    
    names.append("asm_b")
    
    names.append("idm_b")
    
    names.append("energy_b")
    
    names.append("dissimilarity_b")
    
    names.append("mean_reference_b")
    
    names.append("mean_neighbor_b")
    
    names.append("var_refference_b")
    
    names.append("var_neighbor_b")
    
    names.append("correlation_b")
    
    names.append("shade_b")
    
    names.append("prominence_b")
        
    img = cv2.imread(image)
    glcm = create_glcm(img)
    calc_texture_features(glcm,feats)
    
    blue,green,red = cv2.split(img)
    zeros = np.zeros(red.shape,np.uint8)
    red_channel = cv2.merge((zeros,zeros,red))
    glcm = create_glcm_cimage(red_channel,2)
    calc_texture_features(glcm,feats)
    
    zeros = np.zeros(green.shape,np.uint8)
    green_channel = cv2.merge((zeros,green,zeros))
    glcm = create_glcm_cimage(green_channel,1)
    calc_texture_features(glcm,feats)
    
    zeros = np.zeros(blue.shape,np.uint8)
    blue_channel = cv2.merge((blue,zeros,zeros))
    glcm = create_glcm_cimage(blue_channel,0)
    calc_texture_features(glcm,feats)

    df = pd.DataFrame(columns=names)
    sr=pd.Series(feats,index=df.columns)
    df = df.append(sr,ignore_index=True)
    return df

