# deep-learning-hack2018
Contains source code for BFS Fintech hackathon 2018

Please follow the steps below to execute the code

1. Download nnmodel.py and the test, train datasets.
2. Change the location of the test, train datasets in nnmodel.py
3. Open command terminal, go to the directory where nnmodel.py is located.
4. Start python interpreter.
5. Execute the below code
    import nnmodel\
    from nnmodel import X_train, Y_train_hot, X_test, Y_test_hot, model\
    params = model(X_train.T, Y_train_hot, X_test.T, Y_test_hot)\
6. This should run the neural network and print accurracy.\
    Cost after epoch 0: 0.680292\
    Cost after epoch 100: 0.167127\
    Cost after epoch 200: 0.135679\
    Cost after epoch 300: 0.111280\
    Cost after epoch 400: 0.093516\
    Cost after epoch 500: 0.078868\
    Cost after epoch 600: 0.063822\
    Cost after epoch 700: 0.054379\
    Cost after epoch 800: 0.047892\
    Cost after epoch 900: 0.043542\
    
    Parameters have been trained!\
    Train Accuracy: 98.80478382110596\
    Test Accuracy: 97.34748005867004\
