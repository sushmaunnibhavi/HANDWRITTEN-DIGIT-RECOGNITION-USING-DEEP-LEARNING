#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:04:35 2019

@author: sushma
"""
"""
Handwritten digit recognition using convolutional neural networks(CNN)
Here we are traininng a multi level neural network to recognize the digits
Here we are using a database called MNIST which contains 60000 handwritten digits with their labels
After training the neural network,it should now classify the new image correctly


Step 1:download the database of images from MNIST
Step 2:set up a neural network with required number of layers and nodes and train the network
Step 3:feed the training data to the neural network
Step 4:we will check what the output will be like for one image
Step 5:feed the test data set of 10000 images to trained neural network and check its accuracy
"""
import numpy as np
import urllib.request
import gzip
#Step 1:Download files,unzip them and give the training set data as numpy arrays

def load_dataset():
    def download(filename,source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading",filename)
        
        #download the specified file from the url and store it on the disk
        urllib.request.urlretrieve(source+filename,filename) 

    #import library to unzip the zipped files
   
    import os
     #check if file is present on disk,if not present download from website
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        #open the file in binary mode(rb) because zip file is binary file
        with gzip.open(filename,'rb') as f:
            #read data into numpy array using boilerplate wwhich extracts data from zip files

            data=np.frombuffer(f.read(),np.uint8,offset=16)
            """
            this data has 2 issues:(i)it is in the form of 1D array but we want each element of an array to represent an image
            .ie. convert it into an array of images.Done by reshaping.Each image is 28*28 pixels.Image can be monochrome(has only 1 color channel
            representing grayness),or can contain 3 color channels(R,G,B)
            (ii)has to be converted into from byte to float

            we want to reshape into 28*28 arrays.this gives 4 D array 
            1st dimension:number of images 
            2nd dimension:represents number of channels(1 for monochromatic and 3 for fully coloured)
            3rd and 4th dimension represent pixel
            """
            data=data.reshape(-1,1,28,28)
            """here 1st dimension is made -1,the number of images will be inferred from value of other 
            dimensions and length of input array
            """
            #convert byte to float in range (0,1)
            return data/np.float32(256)
    
    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename,'rb') as f:
            #we now have a numpy array of integers,value corresponding to image
            data=np.frombuffer(f.read(),np.uint8,offset=8)
        return data

    X_train=load_mnist_images('train-images-idx3-ubyte.gz')
    y_train=load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test=load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test=load_mnist_images('t10k-labels-idx1-ubyte.gz')

    return X_train,y_train,X_test,y_test

X_train,y_train,X_test,y_test=load_dataset()






