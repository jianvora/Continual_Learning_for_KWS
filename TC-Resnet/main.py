if __name__=='__main__':
    from load_data import load_data_from_folder
    from train import get_tc_resnet_8, get_tc_resnet_14
    from keras.optimizers import Adam
    #from keras.callbacks import ModelCheckpoint

    # Make a bottle neck model
    original_model    = get_tc_resnet_8((321, 40), 30, 1.5) #model corresponding to kws on google speech cmds: input length, num_channel, num_classes
    original_model.load_weights('weights.h5') #Assuming this file is loaded in the current working dir
    bottleneck_input  = original_model.get_layer(index=0).input
    #print(bottleneck_input)
    bottleneck_output = original_model.get_layer(index=-2).output
    #print(bottleneck_output)
    bottleneck_model  = Model(inputs=bottleneck_input,outputs=bottleneck_output)

    # Add the last softmax layer
    for layer in bottleneck_model.layers:
        layer.trainable = False

    new_model = Sequential()
    new_model.add(bottleneck_model)
    kws_classes = 100
    new_model.add(Dense(kws_classes, activation="softmax", input_dim=2808))

    #Split dataset
    X_train, y_train, X_test, y_test, X_validation, y_validation, classes = load_new_data_from_folder('dataset/train') # change this location as per the dir of dataset
    num_classes = len(classes)
    (num_train, input_length, num_channel) = X_train.shape
    num_test = X_test.shape[0]
    num_validation = X_validation.shape[0]
    #print(num_classes)
    #print(num_train)
    #print(num_test)
    #print(num_validation)

    new_model.compile(optimizer=Adam(),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    new_model.fit(x=X_train, y=y_train, batch_size=512, epochs=500, validation_data=(X_test, y_test))
    print(new_model.evaluate(X_validation, y_validation))


