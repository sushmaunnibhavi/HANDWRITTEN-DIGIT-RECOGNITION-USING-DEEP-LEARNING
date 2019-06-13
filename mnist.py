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
            print(data.shape[0])
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
    y_test=load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    return X_train,y_train,X_test,y_test

X_train,y_train,X_test,y_test=load_dataset()

import matplotlib.pyplot as plt
#convert numpy array into image using imshow
plt.show(plt.imshow(X_train[1][0]))
plt.show(plt.imshow(X_train[0][0]))
plt.show(plt.imshow(X_test[0][0]))
#Step2: Set up a neural network and train it

"""
use 2 packages:
    theano :mathematical package that aallows us to work with high dimensional arrays-tensor
    lasagne:it uses theano and helps in building neural networks.Used to set up and train neural network
"""
import lasagne
import theano
import theano.tensor as T
def build_NN(input_var=None):
    """
    we are creating a neural network of 2 hidden layers of 800 nodes each and output
    layer will have 10 nodes,numbered 0-9 and output at each node will be a value between
    0-1.The node with the highest output will be the  predicted output.
    First there is an input layer-expected input shape is 1*28*28 for 1 img
    Now we will link this input layer to input_var(which will be the array of images
    that we will pass later on)
    """
    
    """
    Here we set up input layer l_in and shape of input will be 1*28*28 and link the 
    input to input var
    """
    l_in=lasagne.layers.InputLayer(shape=(None,1,28,28),input_var=input_var)
    
    """
    Since we are using CNN we add a 20% dropout where randomly 20% of the edges
    between the inputs and the next layer will be dropped-done to avoid 
    overfitting
    """
    
    l_in_drop=lasagne.layers.DropoutLayer(l_in,p=0.2)
    
    
    """
    Addd a layer(hidden layer) with 800 nodes ,initially it is dense or fully
    connected.It takes the previous layer(l_in_drop) as input.
    Glorot helps to initialize the layer with some weights,this is done 
    so that training becomes faster
    """
    l_hid1=lasagne.layers.DenseLayer(l_in_drop,num_units=800,
                                     nonlinearity=lasagne.nonlinearities.rectify,
                                     W=lasagne.init.GlorotUniform())
    """
    We will now add a dropout of 50% to first hidden layer
    """
    
    l_hid1_drop=lasagne.layers.DropoutLayer(l_hid1,p=0.5)
    
    """
    Add another hidden layer which takes as its input the first hidden layyer
    """
    l_hid2=lasagne.layers.DenseLayer(l_hid1_drop,num_units=800,
                                     nonlinearity=lasagne.nonlinearities.rectify,
                                     W=lasagne.init.GlorotUniform())
     
    l_hid2_drop=lasagne.layers.DropoutLayer(l_hid2,p=0.5)
     
    """
    Add the final output layer,has 10 nodes,each one for each digit
    """
     
    l_out=lasagne.layers.DenseLayer(l_hid2_drop,num_units=10,
                                     nonlinearity=lasagne.nonlinearities.softmax)
    """
    Output is a softmax where we get outputs between 0-1,max of those
    is the final output
    """
    return l_out
"""
After setting up the network,we have to tell the network how to train itself
.ie. we need to find the values of all the weights.
To do this set up some functions, .ie. initialize some empty arrays which
will act as placeholders for actual training/test data and we will give 
those placeholder data to network(input_var)
"""
    
input_var=T.tensor4('inputs')#an empty 4D array
target_var=T.ivector('targets')#empty 1D array to represent labels
    
network=build_NN(input_var)#call the function that initializes NN
    
    
"""
To train the network we do:
      1.compute an error function
"""
prediction=lasagne.layers.get_output(network)
loss=lasagne.objectives.categorical_crossentropy(prediction,target_var)
"""
Categorical cross entropy is one of the standard error function for
classification problems
"""
loss=loss.mean()
    
    
"""
Tell the network how to update value of weights based  on error
function.
Params contains all the weights of the network currently 
"""
params=lasagne.layers.get_all_params(network,trainable=True)
    
"""
now params will be incrementally changed based on error value
"""
updates=lasagne.updates.nesterov_momentum(loss,params,learning_rate=0.01, momentum=0.9)
"""
Nesterov momentum is provided by lasagne for updating the weights
"""
    
    
"""
we'll use theano to compile a function that is going to represent a single 
training step .ie. compute error,find current weights,update weights
"""
train_fn=theano.function([input_var,target_var],loss,updates=updates)
    
"""
calling this function updates the weights until we get a min value of error
"""
    
"""
Step 3:feed the training data to neural network
let us train it 10 times
"""
num_training_steps=20
for step in range(num_training_steps):
    #pass the training images and labels
    train_err=train_fn(X_train,y_train)
    print('Current step is'+str(step))
    
    
"""
Step 4:Now the neural network has been trained.
We will now check what the output looks like for a test image.
"""
#pass the trained network into a function called test_prediction
test_prediction=lasagne.layers.get_output(network)
#create a function that takes in the input images and predicts the output
val_fn=theano.function([input_var],test_prediction)
    
#pass the first image in test images to val_fn
print(val_fn([X_test[0]]))#gives the output for 1 image
print(y_test[0])  

"""
Now we will feed a dataset of 10000 images and check its accuracy.
Now we create a function that takes in the test images and labels and 
feed it into our network and check the accuracy
""" 
test_prediction=lasagne.layers.get_output(network,deterministic=True)
test_acc=T.mean(T.eq(T.argmax(test_prediction,axis=1),target_var),dtype=theano.config.floatX)
"""
Here to calculate acccuracy,we check the index of the max value in each
test prediction and match it with the actual value
"""
acc_fn=theano.function([input_var,target_var],test_acc)
print(acc_fn(X_test,y_test))

"""
To improve the accuracy:
    1.Can increase the number of training steps
    2.Can divide the training dataset into small subsets and run them 
    individually
    
"""



    
    
    
 
     
    





