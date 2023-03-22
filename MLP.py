# Patel, Meet
# 1002_077_063
# 2023_03_19
# Assignment_02_01

import numpy as np
import tensorflow as tf

def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs=1,loss="svm",
                              validation_split=[0.8,1.0],weights=None,seed=2):
    
    np.random.seed(seed)

    w = []
    # array to hold actual outputs for test
    outputs_final = []
    # list to hold MSE errors for each epoch
    errors_final = []

    def anotherWayToSplitData(X_train,Y_train):
        train_X = X_train[:int(len(X_train)*validation_split[0])]
        test_X = X_train[int(len(X_train)*validation_split[0]):]
        train_y = Y_train[:int(len(Y_train)*validation_split[0])]
        test_y = Y_train[int(len(Y_train)*validation_split[0]):]

    def split_data(X_train, Y_train, split_range=[0.2, 0.7]):
        starting_point = int(split_range[0] * X_train.shape[0])
        end = int(split_range[1] * X_train.shape[0])
        return np.concatenate((X_train[:starting_point], X_train[end:])), np.concatenate(
            (Y_train[:starting_point], Y_train[end:])), X_train[starting_point:end], Y_train[starting_point:end]
    
    def batch_generator(X, y, batch_size=32):
        for i in range(0, X.shape[0], batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]
        if X.shape[0] % batch_size != 0:
            yield X[-(X.shape[0] % batch_size):], y[-(X.shape[0] % batch_size):]

    X_tr, y_tr, X_val, y_val = split_data(X_train, Y_train, split_range=validation_split)
    print(len(X_tr),X_tr)
    print(len(X_val),X_val)
    print(len(y_tr))
    print(len(y_val))

    np.random.seed(seed)
    if weights!=None:
        w = weights
    else:
        for i in range(len(layers)):
            np.random.seed(seed)
            if i == 0:
                w.append(tf.Variable(np.float32(np.random.randn(X_tr.shape[1]+1,layers[i]))))
            else:
                w.append(tf.Variable(np.float32(np.random.randn(layers[i-1]+1, layers[i]))))

    # Predict Function
    def predict(input_X):
        for each_layer in range(len(layers)):
            # add ones to input X as features
            input_X = tf.concat([tf.ones((tf.shape(input_X)[0], 1)),input_X], axis=1)
            input_X = tf.matmul(input_X, w[each_layer])
            if activations[each_layer].lower() == "linear":
                input_X = input_X
            if activations[each_layer].lower() == "relu":
                input_X = tf.nn.relu(input_X)
            if activations[each_layer].lower() == "sigmoid":
                input_X = tf.nn.sigmoid(input_X)
        return input_X

    def loss_function(predicted,actual, loss_variable):
        if loss_variable == "mse":
            print("mse")
            return tf.reduce_mean(tf.square(actual - predicted))
        if loss_variable == "svm":
            print("svm")
            intermediate = tf.maximum(0, 1 - tf.multiply(predicted, actual))
            return tf.reduce_mean(intermediate)
        if loss_variable == "cross_entropy":
            print("cross entropy")
            intermediate =  tf.nn.softmax_cross_entropy_with_logits(labels=actual,logits=predicted)
            return tf.reduce_mean(intermediate)

    def anotherWayToCreateBatchNotUsed(train_X,train_y):
        for starting_point_index in range(0,len(train_X),batch_size):
            batch_step = starting_point_index + batch_size
            if batch_step>len(train_X):
                batch_step = len(train_X)
            x_input_batch = tf.Variable(train_X[starting_point_index:batch_step,:])
            y_input_batch = tf.Variable(train_y[starting_point_index:batch_step,:])

    for epoch in range(epochs):
        for X_batch, y_batch in batch_generator(X_tr, y_tr, batch_size=batch_size):
            with tf.GradientTape() as tape:
                net = predict(X_batch)
                loss_calculated = loss_function(net, y_batch, loss.lower())
                weight_gradient = tape.gradient(loss_calculated, w)
            # print("W ", w)

                print("weight_gradient",weight_gradient)
            # upgrade gradient after each batch
                for each in range(len(w)):
                    w[each].assign_sub(alpha * weight_gradient[each])

        # Calculate Validation loss after each epoch
        predicted_testx = predict(X_val)
        loss_calculated = loss_function(predicted_testx,y_val, loss.lower())
        errors_final.append(loss_calculated)
    
    # Calulate final validation error on test data
    net = predict(X_val)
    # print("Final_net : ", net.shape)

    # # print("W",w)
    # print(len(errors_final))
    # print("error",errors_final)
    # print("op",net)
    return w,errors_final,net


    # This function creates and trains a multi-layer neural Network
    # X_train: numpy array of input for training [nof_train_samples,input_dimensions]
    # Y_train: numpy array of desired outputs for training samples [nof_train_samples,output_dimensions]
    # layers: list of integers representing number of nodes in each layer
    # activations: list of case-insensitive activations strings corresponding to each layer. The possible activations
    # are, "linear", "sigmoid", "relu".
    # alpha: learning rate
    # epochs: number of epochs for training.
    # loss: is a case-insensitive string determining the loss function. The possible inputs are: "svm" , "mse",
    # "cross_entropy". for cross entropy use the tf.nn.softmax_cross_entropy_with_logits().
    # validation_split: a two-element list specifying the normalized starting_point and end point to
    # extract validation set. Use floor in case of non integers.
    # weights: list of numpy weight matrices. If weights is equal to None then it should be ignored, otherwise,
    # the weight matrices should be initialized by the values given in the weight list (no random
    # initialization when weights is not equal to None).
    # seed: random number generator seed for initializing the weights.
    # return: This function should return a list containing 3 elements:
        # The first element of the return list should be a list of weight matrices.
        # Each element of the list should be a 2-d numpy array which corresponds to the weight matrix of the
        # corresponding layer.

        # The second element should be a one dimensional numpy array of numbers
        # representing the error after each epoch. Each error should
        # be calculated by using the validation set while the network is frozen.
        # Frozen means that the weights should not be adjusted while calculating the error.

        # The third element should be a two-dimensional numpy array [nof_validation_samples,output_dimensions]
        # representing the actual output of the network when validation set is used as input.

    # Notes:
    # The data set in this assignment is the transpose of the data set in assignment_01. i.e., each row represents
    # one data sample.
    # The weights in this assignment are the transpose of the weights in assignment_01.
    # Each output weights in this assignment is the transpose of the output weights in assignment_01
    # DO NOT use any other package other than tensorflow and numpy
    # Bias should be included in the weight matrix in the first row.
    # Use steepest descent for adjusting the weights
    # Use minibatch to calculate error and adjusting the weights
    # Reseed the random number generator when initializing weights for each layer.
    # Use numpy for weight to initialize weights. Do not use tensorflow weight initialization.
    # Do not use any random method from tensorflow
    # Do not shuffle data
    # i.e., Initialize the weights for each layer by:
    # np.random.seed(seed)
    # np.random.randn()

    pass