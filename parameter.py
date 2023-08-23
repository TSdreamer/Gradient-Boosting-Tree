import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from pymatgen import Composition
import seaborn as sns
from scipy import stats,integrate
from matminer.featurizers import composition as cf
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementFraction
from sklearn import feature_selection
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_curve, auc  ###TO CALCULATE ROC AND AUC
from sklearn.model_selection import KFold

import pickle

path = r"data/Cleaned_Data0803.xlsx"
model_path = r"model/gradientboostingtree.pickle"
feature_calculators = MultipleFeaturizer([cf.Stoichiometry(),
                                          cf.ElementProperty.from_preset("magpie"),
                                          cf.ValenceOrbital(props=['avg']),
                                          cf.IonProperty(fast=True)])
labels = ["0", "1", "2"]


# import the data
def load_need_dataSet(path, col_idx=0):
    df = pd.read_excel(path, usecols=range(0, 8))
    # df = load_dataset("brgoch_superhard_training")
    columns = df.columns.tolist()
    # print(columns[3])
    se = df[columns[3]]
    label = df[columns[-1]]
    return se, columns[3], label


# transfer the formula
def forlumaToFeature(se):
    out = []
    feature_labels = feature_calculators.feature_labels()
    for idx, value in se.items():
        print("*****************{}:{}*******".format(idx, value))
        forluma = Composition(value)
        print(forluma)
        tmp = feature_calculators.featurize(forluma)
        out.append(tmp)
    return np.array(out), feature_labels


# extract the formula
def getValidFeature(featureArr, label, y, feature_num=15):
    # via Select_model to choose feature
    valid_feature = SelectKBest(chi2, k=feature_num).fit_transform(featureArr, y)
    idx = SelectKBest(chi2, k=feature_num).fit(featureArr, y).get_support()
    print(valid_feature.shape)
    valid_label = []
    valid_idx = []
    for ii, lab in enumerate(label):
        if idx[ii]:
            valid_label.append(lab)
            valid_idx.append(ii)
    print(valid_idx)
    return valid_feature, valid_label, valid_idx


# plot the confusion matrix to visulize
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.jpg")
    plt.show()


# train the model
def bulid_model(X, y, feature_label):
    # to train and to split the dataset into several parts
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

    # to build the hyperparameters of the code
    param_dict = {"n_estimators": [20, 40, 60, 80, 100, 120], "learning_rate":[0.2,0.5,0.8,1],"max_depth": [10, 20, 30], "min_samples_leaf": [1, 2, 3],"subsample":[0.5,0.6,0.7,0.8,0.9,1]}
    rf = GradientBoostingClassifier()
    # using the gridsearch to find the best parameters
    estimator = GridSearchCV(rf, param_grid=param_dict, cv=10)
    # 10 fold cross validation
    # kf = KFold(n_splits=10)
    # for train_idx, test_idx in kf.split(X):
    # x_train = X[train_idx]
    # y_train = y[train_idx]
    # x_test = X[test_idx]
    # y_test = y[test_idx]

    estimator.fit(x_train, y_train)
    score = estimator.score(X, y)
    print("the score of the trainsets is:", score)
    # score_test = rf.score(x_test, y_test)
    # print("the score of test-train sets is:", score_test)
    print("the best result in the cross-validations is:\n", estimator.best_score_)
    print("the best parameter model is:\n", estimator.best_estimator_)
    print("every accuracy of the cross-validation is:\n", estimator.cv_results_)
    # to print the importance of each features
    print(estimator.best_estimator_.feature_importances_)

    fw = open(model_path, 'wb')
    pickle.dump(estimator.best_estimator_, fw)
    fw.close()



    score_tol = estimator.best_estimator_.score(X, y)
    print("the score of the model is:", score_tol)
    # to calculate the confusion matrix
    y_hat = estimator.best_estimator_.predict(X)
    confu_mat = confusion_matrix(y, y_hat)
    print(confu_mat)
    # to print the report part
    report = classification_report(y, y_hat)
    print(report)

    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = confu_mat.astype('float') / confu_mat.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')


# to predict the model
def test_model(X, model_path):
    print("*********import the model to make the prediction**********")

    fr = open(model_path, 'rb')
    rf = pickle.load(fr)

    y_hat = rf.predict(X)
    print(y_hat)
    x = np.arange(1, len(y_hat) + 1)

    plt.figure(1)
    sns.distplot(y_hat, bins=3, kde=True, label=['label_0', 'label_1', 'label_2'])
    plt.savefig("hist.jpg")
    plt.show()

    return y_hat


def test_allelement(idx, model_path):
    data = pd.read_excel("element.xlsx")
    element_fraction_labels = data.values[:, 0].tolist()
    # print(len(element_fraction_labels))
    # print(element_fraction_labels)
    out_ele = []
    for i in range(len(element_fraction_labels)):
        for j in range(i + 1, len(element_fraction_labels)):
            name1 = element_fraction_labels[i] + element_fraction_labels[j]
            name2 = element_fraction_labels[1] + '2' + element_fraction_labels[j]
            name3 = element_fraction_labels[1] + element_fraction_labels[j] + '2'
            out_ele.append(name1)
            out_ele.append(name2)
            out_ele.append(name3)

    out_se = pd.Series(out_ele)

    feature, feature_label = forlumaToFeature(out_se)

    valid_feature = feature[:, idx]
    valid_feature = np.nan_to_num(valid_feature)
    y_predict = test_model(valid_feature, model_path)
    out_dict = {}
    out_dict['compose'] = out_ele
    out_dict['label'] = y_predict.tolist()

    df = pd.DataFrame(out_dict)
    df.to_csv("result_GBT.csv")
    plt.figure(figsize=(10,8))
    plt.scatter(out_ele[:30], y_predict.tolist()[:30], c=y_predict.tolist()[:30])
    plt.xlabel("element name")
    plt.ylabel('label')
    plt.savefig("scatter.jpg")
    plt.show()


if __name__ == '__main__':
    # import the data
    series_1, column_name, y = load_need_dataSet(path, 3)
    # transfer the features
    feature, feature_label = forlumaToFeature(series_1)

    # use all the features to train the model
    bulid_model(feature, y, feature_label)

    # to extract the features
    valid_feature, valid_label, idx = getValidFeature(feature, feature_label, y, 20)

    print("the remaining features are:", valid_label)

    # to train the model
    bulid_model(valid_feature, y, valid_label)
    # to predict the model
    test_model(valid_feature, model_path)

    test_allelement(idx, model_path)
