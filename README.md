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
    params = model(X_train.T, Y_train_hot, X_test.T, Y_test_hot)
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
    Cost after epoch 900: 0.043542
    
    Parameters have been trained!\
    Train Accuracy: 98.80478382110596\
    Test Accuracy: 97.34748005867004
# Credit Scoring
1. pip install scorecardpy
2. replace the files under <venv_name>Lib/site-packages/scorecardpy/ with the files in the repository under scorecardpy.
3. Open python interpreter and execute the following commands.\
import scorecardpy as sc\
dat, train_dat, test_dat = sc.loaddata()\
break_list = {\
    ' Days Receivable' : [-0.08664425, -0.06541378, -0.05153443, -0.03674076],\
    ' Current Ratio' : [-0.0640368 , -0.04113432, -0.02835715, -0.01063787],\
    ' Cash to TA' : [-0.8514839 , -0.73214835, -0.49347723, -0.14944679,  0.53710896],\
    ' Working Capital To TA' : [-5.57055097, -0.68866573, -0.45836958, -0.0088759 ,  0.71993279],\
    ' Asset Turnover' : [-2.76260647, -0.74542403, -0.49327622, -0.14026929,  0.61617413],\
    ' Gross Profit Margin' : [-5.98486371, -0.84459514, -0.42485723,  0.09829436,  1.46700493],\
    'Operating Profit Margin' : [-30.57372395,  -0.27735587,  -0.12499386,   0.06148567, 0.34109385],\
    ' Net Profit Margin' : [-26.51417009,  -0.15441278,  -0.03931382,   0.16456454],\
    ' Debt to asset' : [-1.40499214, -0.86038468, -0.3268316 ,  0.14053879,  0.68371061],\
    ' Debt to equity' : [-34.59905207,   0.01828744,   0.0282831 ,   0.04440189],\
    ' Equity multiplier' : [-34.099279  ,   0.02001166,   0.03155167,   0.0492223],\
    'Time interest earned coverage': [-20.68392678,  -0.07735931,  -0.06228727,  -0.02296753],\
    ' Operating return on asset' : [-16.42388261,  -0.38238724,  -0.12272275,   0.25555392],\
    ' Return on asset' : [-16.68367281,  -0.36116653,  -0.10361219,   0.24694788],\
    ' Return on Equity' : [-38.71834141,  -0.00084629,   0.01014875,   0.02369991, 0.04708082],\
    ' EBITDA to TA' : [-12.91160461,  -0.3941609 ,  -0.15583658,   0.23008634],\
    ' Cashflow to expenditure' : [-3.36329427, -0.03850119],\
    'Cashflow to debt' : [-7.26605425, -0.0769386],\
}\
bins_adj = sc.woebin(dat, y="Result", breaks_list=break_list, no_cores=1)\
train_woe = sc.woebin_ply(train_dat, bins_adj, no_cores = 1)\
test_woe = sc.woebin_ply(test_dat, bins_adj, no_cores = 1)\
y_train = train_woe.loc[:,'Result']\
X_train = train_woe.loc[:,train_woe.columns != 'Result']\
y_test = test_woe.loc[:,'Result']\
X_test = test_woe.loc[:,test_woe.columns != 'Result']\
from sklearn.linear_model import LogisticRegression\
lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)\
lr.fit(X_train, y_train)\
card = sc.scorecard(bins_adj, lr, X_train.columns, odds0=1/9, pdo=200)\
train_score = sc.scorecard_ply(train_dat, card, print_step=0)\
test_score = sc.scorecard_ply(test_dat, card, print_step=0)\


