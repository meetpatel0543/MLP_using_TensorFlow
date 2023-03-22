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
