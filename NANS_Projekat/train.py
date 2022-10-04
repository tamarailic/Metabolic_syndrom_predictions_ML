import numpy as np
import pandas as pd
from lr import *
from dt import *
from rf import *
from knn import *
from svm import *
import pickle as pc

def train_test_split(df,predictor_col,test_size=0.1,random_state=1):
    df=df.sample(n=df.shape[0],random_state=random_state)
    partition=int(df.shape[0]*test_size)
    X,y=df.drop(columns=[predictor_col]),df[predictor_col]
    return np.array(X[:df.shape[0]-partition]),np.array(X[df.shape[0]-partition:]),np.array(y[:df.shape[0]-partition]),np.array(y[df.shape[0]-partition:])

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)

def rescale(X):
    mean = np.mean(X)
    std = np.std(X)
    scaled_X = [(i - mean) / std for i in X]
    return scaled_X

def rescale_arr(arr):
    arr1 = arr
    for i in range(arr1.shape[1]):
        arr1[:][i] = rescale(arr1[:][i])

    return arr1

def lr_train():
    learning_rates = [0.001, 0.003, 0.005, 0.01, 0.015]
    best_acc = 0
    best_acc_sci = 0
    best_acc_stat = 0
    lr_best = None
    lr_best_sci = None
    lr_best_stat = None
    acc_test = []
    acc_test_sci = []
    acc_test_stat = []
    for i in learning_rates:
        print("learning rate is: ", i)
        print("-------------------------------------------------------")
        models = LogisticRegression()
        models.fit(X_train_std, y_train, gamma=i, iterations=100)

        predictions_lr_train = models.predict(X_train_std)
        print("Logistic_regression train accuracy: ", round(accuracy(y_train, predictions_lr_train) * 100), "%")

        predictions_lr_test = models.predict(X_test_std)
        print("Logistic_regression test accuracy: ", round(accuracy(y_test, predictions_lr_test) * 100), "%")
        print("-------------------------------------------------------")
        acc_test.append(accuracy(y_test, predictions_lr_test))
        if accuracy(y_test, predictions_lr_test) > best_acc:
            best_acc = accuracy(y_test, predictions_lr_test)
            lr_best = models

        print("Science")
        models_sci = LogisticRegression()
        models_sci.fit(X_train_std_sci, y_train_sci, gamma=i, iterations=100)

        predictions_lr_train = models_sci.predict(X_train_std_sci)
        print("Logistic_regression train accuracy: ", round(accuracy(y_train_sci, predictions_lr_train) * 100), "%")

        predictions_lr_test = models_sci.predict(X_test_std_sci)
        print("Logistic_regression test accuracy: ", round(accuracy(y_test_sci, predictions_lr_test) * 100), "%")
        print("-------------------------------------------------------")
        acc_test_sci.append(accuracy(y_test_sci, predictions_lr_test))
        if accuracy(y_test_sci, predictions_lr_test) > best_acc_sci:
            best_acc_sci = accuracy(y_test_sci, predictions_lr_test)
            lr_best_sci = models_sci

        print("Stats")
        models_stat = LogisticRegression()
        models_stat.fit(X_train_std_stat, y_train_stat, gamma=i, iterations=100)

        predictions_lr_train = models_stat.predict(X_train_std_stat)
        print("Logistic_regression train accuracy: ", round(accuracy(y_train_stat, predictions_lr_train) * 100), "%")

        predictions_lr_test = models_stat.predict(X_test_std_stat)
        print("Logistic_regression test accuracy: ", round(accuracy(y_test_stat, predictions_lr_test) * 100), "%")
        print("-------------------------------------------------------")
        print("\n")
        acc_test_stat.append(accuracy(y_test_stat, predictions_lr_test))
        if accuracy(y_test_stat, predictions_lr_test) > best_acc_stat:
            best_acc_stat = accuracy(y_test_stat, predictions_lr_test)
            lr_best_stat = models_stat

    save_lr_models(lr_best,lr_best_sci,lr_best_stat)

def dt_train():
    max_depths = [5, 10, 15, 25, 50, 75, 100, 150, 250]
    best_acc = 0
    best_acc_sci = 0
    best_acc_stat = 0
    tree_best = None
    tree_best_sci = None
    tree_best_stat = None
    acc_trees_test = []
    acc_trees_test_sci = []
    acc_trees_test_stat = []
    for i in max_depths:
        print("Max depth is: ", i)
        print("-------------------------------------------------------")
        trees = DecisionTree(max_depth=i)
        trees.fit(X_train_std, y_train)
        predictions_train = trees.predict(X_train_std)
        print("Decision tree whole dataset train accuracy: ", round(accuracy(y_train, predictions_train) * 100), "%")

        predictions_test = trees.predict(X_test_std)
        print("Decision tree whole dataset test accuracy: ", round(accuracy(y_test, predictions_test) * 100), "%")
        print("-------------------------------------------------------")
        acc_trees_test.append(accuracy(y_test, predictions_test))
        if accuracy(y_test, predictions_test) > best_acc:
            best_acc = accuracy(y_test, predictions_test)
            tree_best = trees

        print("Science")
        trees_sci = DecisionTree(max_depth=i)
        trees_sci.fit(X_train_std_sci, y_train_sci)

        predictions_train = trees_sci.predict(X_train_std_sci)
        print("Decision tree science dataset train accuracy: ", round(accuracy(y_train_sci, predictions_train) * 100),
              "%")

        predictions_test = trees_sci.predict(X_test_std_sci)
        print("Decision tree science dataset test accuracy: ", round(accuracy(y_test_sci, predictions_test) * 100), "%")
        print("-------------------------------------------------------")
        acc_trees_test_sci.append(accuracy(y_test_sci, predictions_test))
        if accuracy(y_test_sci, predictions_test) > best_acc_sci:
            best_acc_sci = accuracy(y_test_sci, predictions_test)
            tree_best_sci = trees_sci

        print("Stats")
        trees_stat = DecisionTree(max_depth=i)
        trees_stat.fit(X_train_std_stat, y_train_stat)

        predictions_train = trees_stat.predict(X_train_std_stat)
        print("Decision tree stat dataset train accuracy: ", round(accuracy(y_train_stat, predictions_train) * 100),
              "%")

        predictions_test = trees_stat.predict(X_test_std_stat)
        print("Decision tree stat dataset test accuracy: ", round(accuracy(y_test_stat, predictions_test) * 100), "%")
        print("-------------------------------------------------------")
        print("\n")
        acc_trees_test_stat.append(accuracy(y_test_stat, predictions_test))
        if accuracy(y_test_stat, predictions_test) > best_acc_stat:
            best_acc_stat = accuracy(y_test_stat, predictions_test)
            tree_best_stat = trees_stat
    save_dt_models(tree_best,tree_best_sci,tree_best_stat)

def rf_train():
    n_trees = [3, 5, 7, 10, 12, 15, 25, 50]
    best_acc = 0
    best_acc_sci = 0
    best_acc_stat = 0
    forest_best = None
    forest_best_sci = None
    forest_best_stat = None
    acc_forests_test = []
    acc_forests_test_sci = []
    acc_forests_test_stat = []
    for i in n_trees:
        print("Number of trees in forest is: ", i)
        print("-------------------------------------------------------")
        forests = RandomForest(n_trees=i)
        forests.fit(X_train_std, y_train)
        predictions_train = forests.predict(X_train_std)
        print("RandomForest whole dataset train accuracy: ", round(accuracy(y_train, predictions_train) * 100), "%")

        predictions_test = forests.predict(X_test_std)
        print("RandomForest whole dataset test accuracy: ", round(accuracy(y_test, predictions_test) * 100), "%")
        print("-------------------------------------------------------")
        acc_forests_test.append(accuracy(y_test, predictions_test))
        if accuracy(y_test, predictions_test) > best_acc:
            best_acc = accuracy(y_test, predictions_test)
            forest_best = forests

        print("Science")
        forests_sci = RandomForest(n_trees=i)
        forests_sci.fit(X_train_std_sci, y_train_sci)

        predictions_train = forests_sci.predict(X_train_std_sci)
        print("RandomForest science dataset train accuracy: ", round(accuracy(y_train_sci, predictions_train) * 100),
              "%")

        predictions_test = forests_sci.predict(X_test_std_sci)
        print("RandomForest science dataset test accuracy: ", round(accuracy(y_test_sci, predictions_test) * 100), "%")
        print("-------------------------------------------------------")
        acc_forests_test_sci.append(accuracy(y_test_sci, predictions_test))
        if accuracy(y_test_sci, predictions_test) > best_acc_sci:
            best_acc_sci = accuracy(y_test_sci, predictions_test)
            forest_best_sci = forests_sci

        print("Stats")
        forests_stat = RandomForest(n_trees=i)
        forests_stat.fit(X_train_std_stat, y_train_stat)

        predictions_train = forests_stat.predict(X_train_std_stat)
        print("RandomForest stat dataset train accuracy: ", round(accuracy(y_train_stat, predictions_train) * 100), "%")

        predictions_test = forests_stat.predict(X_test_std_stat)
        print("RandomForest stat dataset test accuracy: ", round(accuracy(y_test_stat, predictions_test) * 100), "%")
        print("-------------------------------------------------------")
        print("\n")
        acc_forests_test_stat.append(accuracy(y_test_stat, predictions_test))
        if accuracy(y_test_stat, predictions_test) > best_acc_stat:
            best_acc_stat = accuracy(y_test_stat, predictions_test)
            forest_best_stat = forests_stat
    save_rf_models(forest_best,forest_best_sci,forest_best_stat)

def knn_train():
    k = [3, 5, 7, 10, 12, 15, 25, 50, 100]
    best_acc = 0
    best_acc_sci = 0
    best_acc_stat = 0
    knns_best = None
    knns_best_sci = None
    knns_best_stat = None
    acc_knns_test = []
    acc_knns_test_sci = []
    acc_knns_test_stat = []
    for i in k:
        print("Number of neighbors is: ", i)
        print("-------------------------------------------------------")
        knns = KNN(k=i)
        knns.fit(X_train_std, y_train)
        predictions_train = knns.predict(X_train_std)
        print("KNN whole dataset train accuracy: ", round(accuracy(y_train, predictions_train) * 100), "%")

        predictions_test = knns.predict(X_test_std)
        print("KNN whole dataset test accuracy: ", round(accuracy(y_test, predictions_test) * 100), "%")
        print("-------------------------------------------------------")
        acc_knns_test.append(accuracy(y_test, predictions_test))
        if accuracy(y_test, predictions_test) > best_acc:
            best_acc = accuracy(y_test, predictions_test)
            knns_best = knns

        print("Science")
        knns_sci = KNN(k=i)
        knns_sci.fit(X_train_std_sci, y_train_sci)

        predictions_train = knns_sci.predict(X_train_std_sci)
        print("KNN science dataset train accuracy: ", round(accuracy(y_train_sci, predictions_train) * 100), "%")

        predictions_test = knns_sci.predict(X_test_std_sci)
        print("KNN science dataset test accuracy: ", round(accuracy(y_test_sci, predictions_test) * 100), "%")
        print("-------------------------------------------------------")
        acc_knns_test_sci.append(accuracy(y_test_sci, predictions_test))
        if accuracy(y_test_sci, predictions_test) > best_acc_sci:
            best_acc_sci = accuracy(y_test_sci, predictions_test)
            knns_best_sci = knns_sci

        print("Stats")
        knns_stat = KNN(k=i)
        knns_stat.fit(X_train_std_stat, y_train_stat)

        predictions_train = knns_stat.predict(X_train_std_stat)
        print("KNN stat dataset train accuracy: ", round(accuracy(y_train_stat, predictions_train) * 100), "%")

        predictions_test = knns_stat.predict(X_test_std_stat)
        print("KNN stat dataset test accuracy: ", round(accuracy(y_test_stat, predictions_test) * 100), "%")
        print("-------------------------------------------------------")
        print("\n")
        acc_knns_test_stat.append(accuracy(y_test_stat, predictions_test))
        if accuracy(y_test_stat, predictions_test) > best_acc_stat:
            best_acc_stat = accuracy(y_test_stat, predictions_test)
            knns_best_stat = knns_stat
    save_knn_models(knns_best,knns_best_sci,knns_best_stat)

def svm_train():
    learning_rates = [0.1, 0.3, 0.5, 0.1, 0.5]
    lambda_params = [0.01, 0.1, 0.15]
    iters = [2, 3, 5, 7, 10, 15]
    acc_svm_test = []
    acc_svm_test_sci = []
    acc_svm_test_stat = []
    best_acc = 0
    best_acc_sci = 0
    best_acc_stat = 0
    best_model = None
    best_model_sci = None
    best_model_stat = None
    y_train_svm = y_train
    y_train_svm_sci = y_train_sci
    y_train_svm_stat = y_train_stat
    y_test_svm = y_test
    y_test_svm_sci = y_test_sci
    y_test_svm_stat = y_test_stat
    for i in learning_rates:
        for j in lambda_params:
            for it in iters:
                print("learning rate is: ", i)
                print("lambda is: ", j)
                print("Number of iterations is: ", it)
                print("-------------------------------------------------------")
                svm = SVM(learning_rate=i, lambda_param=j, n_iters=it)
                svm.fit(X_train_std, y_train_svm)

                predictions_train = svm.predict(X_train_std)
                print("SVM train accuracy: ", round(accuracy(y_train_svm, predictions_train) * 100), "%")

                predictions_test = svm.predict(X_test_std)
                print("SVM test accuracy: ", round(accuracy(y_test_svm, predictions_test) * 100), "%")
                print("-------------------------------------------------------")
                acc_svm_test.append(accuracy(y_test_svm, predictions_test))
                if accuracy(y_test_svm, predictions_test) > best_acc:
                    best_acc = accuracy(y_test_svm, predictions_test)
                    best_model = svm

                print("Science")
                svm_sci = SVM(learning_rate=i, lambda_param=j, n_iters=it)
                svm_sci.fit(X_train_std_sci, y_train_svm_sci)

                predictions_train = svm_sci.predict(X_train_std_sci)
                print("SVM train accuracy: ", round(accuracy(y_train_svm_sci, predictions_train) * 100), "%")

                predictions_test = svm_sci.predict(X_test_std_sci)
                print("SVM test accuracy: ", round(accuracy(y_test_svm_sci, predictions_test) * 100), "%")
                print("-------------------------------------------------------")
                acc_svm_test_sci.append(accuracy(y_test_svm_sci, predictions_test))
                if accuracy(y_test_svm_sci, predictions_test) > best_acc_sci:
                    best_acc_sci = accuracy(y_test_svm_sci, predictions_test)
                    best_model_sci = svm_sci

                print("Stats")
                svm_stat = SVM(learning_rate=i, lambda_param=j, n_iters=it)
                svm_stat.fit(X_train_std_stat, y_train_svm_stat)

                predictions_train = svm_stat.predict(X_train_std_stat)
                print("SVM train accuracy: ", round(accuracy(y_train_svm_stat, predictions_train) * 100), "%")

                predictions_test = svm_stat.predict(X_test_std_stat)
                print("SVM test accuracy: ", round(accuracy(y_test_svm_stat, predictions_test) * 100), "%")
                print("-------------------------------------------------------")
                print("\n")
                acc_svm_test_stat.append(accuracy(y_test_svm_stat, predictions_test))
                if accuracy(y_test_svm_stat, predictions_test) > best_acc_stat:
                    best_acc_stat = accuracy(y_test_svm_stat, predictions_test)
                    best_model_stat = svm_stat
    save_svm_models(best_model,best_model_sci,best_model_stat)

def save_lr_models(lr_best, lr_best_sci, lr_best_stat):
    with open("data/lr_best.pickle","wb") as file:
        pc.dump(lr_best,file)
    with open("data/lr_best_sci.pickle","wb") as file:
        pc.dump(lr_best_sci,file)
    with open("data/lr_best_stat.pickle","wb") as file:
        pc.dump(lr_best_stat,file)

def save_dt_models(tree_best,tree_best_sci,tree_best_stat):
    with open("data/tree_best.pickle","wb") as file:
        pc.dump(tree_best,file)
    with open("data/tree_best_sci.pickle","wb") as file:
        pc.dump(tree_best_sci,file)
    with open("data/tree_best_stat.pickle","wb") as file:
        pc.dump(tree_best_stat,file)

def save_rf_models(forest_best,forest_best_sci,forest_best_stat):
    with open("data/forest_best.pickle","wb") as file:
        pc.dump(forest_best,file)
    with open("data/forest_best_sci.pickle","wb") as file:
        pc.dump(forest_best_sci,file)
    with open("data/forest_best_stat.pickle","wb") as file:
        pc.dump(forest_best_stat,file)

def save_knn_models(knns_best,knns_best_sci,knns_best_stat):
    with open("data/knns_best.pickle","wb") as file:
        pc.dump(knns_best,file)
    with open("data/knns_best_sci.pickle","wb") as file:
        pc.dump(knns_best_sci,file)
    with open("data/knns_best_stat.pickle","wb") as file:
        pc.dump(knns_best_stat,file)

def save_svm_models(best_model,best_model_sci,best_model_stat):
    with open("data/svm_best.pickle","wb") as file:
        pc.dump(best_model,file)
    with open("data/svm_best_sci.pickle","wb") as file:
        pc.dump(best_model_sci,file)
    with open("data/svm_best_stat.pickle","wb") as file:
        pc.dump(best_model_stat,file)

if __name__ == '__main__':
    df = pd.read_csv('metabolicki_sindrom.csv', encoding='unicode_escape')
    df = df.drop(columns=['IME', 'PREZIME'])
    df["OS (percentil)"] = df["OS (percentil)"].astype('category')
    df["OS (percentil)"] = df["OS (percentil)"].cat.codes
    X_train, X_test, y_train, y_test = train_test_split(df, 'METABOLI?KI SINDROM', test_size=0.2, random_state=1)
    X_train_std = rescale_arr(X_train)
    X_test_std = rescale_arr(X_test)

    df_sci = df[['OS(cm)', 'TRIGLICERIDI', 'SISTOLNI PRITISAK', 'DIJASTOLNI PRITISAK', 'HDL', 'E?ERNA BOLEST',
                 'METABOLI?KI SINDROM']]
    X_train, X_test, y_train_sci, y_test_sci = train_test_split(df_sci, 'METABOLI?KI SINDROM', test_size=0.2, random_state=1)

    X_train_std_sci = rescale_arr(X_train)
    X_test_std_sci = rescale_arr(X_test)

    df_stat = df[['UZRAST', 'HIPERTENZIJA (R)', 'OS(cm)', 'TRIGLICERIDI', 'SISTOLNI PRITISAK', 'HDL', 'E?ERNA BOLEST',
                  'METABOLI?KI SINDROM']]
    X_train, X_test, y_train_stat, y_test_stat = train_test_split(df_stat, 'METABOLI?KI SINDROM', test_size=0.2, random_state=1)

    X_train_std_stat = rescale_arr(X_train)
    X_test_std_stat = rescale_arr(X_test)

    lr_train()
    dt_train()
    rf_train()
    knn_train()
    svm_train()
