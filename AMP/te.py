from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import hamming_loss

import warnings
warnings.filterwarnings('ignore')


def multilLabel():
    # 多标签多分类原始标签
    y = [[2,3,4],[2],[0,1,3],[0,1,2,3,4],[0,1,2]]
    # 对标签进行预处理
    mb = MultiLabelBinarizer()
    # y_mb变成N*K的矩阵(N:样本数,K:类别数)
    y_mb = mb.fit_transform(y)

    # 多类别学习,标签形如[0,0,1,1,2,2]
    data = load_iris()
    X,y = data.data,data.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    binary_model = SVC(kernel="linear",random_state=1)
    # one-vs-all形式(既可以多类别问题也可以多标签多分类问题，fir(X,y)中y.shape=[samples] or [samples,classes])
    multi_model = OneVsRestClassifier(binary_model).fit(X_train,y_train)
    # one-vs-one形式(只能用于多类别问题，fit(X,y)函数要求y.shape=[samples])
    #multi_model = OneVsOneClassifier(binary_model).fit(X_train,y_train)
    y_pred = multi_model.predict(X_test)
    print("True Labels:   ",y_test)
    print("Predict Labels:",y_pred)
    print("Accuracy: ",accuracy_score(y_test,y_pred))

    # 多标签多分类
    ml_X,ml_y = make_multilabel_classification()
    print("多标签多分类训练标签:\n",ml_y[:5])
    ml_X_train,ml_X_test,ml_y_train,ml_y_test = train_test_split(ml_X,ml_y,test_size=0.1)
    # one-vs-all
    clf = OneVsRestClassifier(SVC(kernel="linear"))
    clf.fit(ml_X_train,ml_y_train)
    pred_y = clf.predict(ml_X_test)
    print("True Labels:  \n",ml_y_test)
    print("Predict Labels:\n",pred_y)

    print("Hamming_loss: ",hamming_loss(ml_y_test,pred_y))
    print("Accuracy:     ",accuracy_score(ml_y_test,pred_y))


multilLabel()