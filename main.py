#-*- coding:utf-8 -*-
"""

    描述：对提取的特征进行分类器训练和模型的保存
    其他：相关软件版本
          matplotlib:  3.1.3  
          numpy: 1.18.1  
          scipy: 1.4.1 
          pandas: 1.0.3
          sklearn: 0.22.1 
          joblib: 0.14.1
"""
import joblib  # -> 用来保存模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import os
import matplotlib.pyplot as plt

from utils.augment import preprocess
from utils.feature import extract_feature

# -1- 载入数据
path = r"/Users/yuketu/Downloads/data/0HP"
data_mark = "FE"
len_data = 1024
overlap_rate = 50      # -> 50%
random_seed = 1 
fs = 12000

X, y = preprocess(path, 
                    data_mark, 
                    fs, 
                    len_data/fs, 
                    overlap_rate, 
                    random_seed ) 

# -2- 提取特征
FX, Fy = extract_feature(X, y, fs)

# -3- 数据集划分
x_train, x_test, y_train, y_test = train_test_split(FX, 
                                                    Fy, 
                                                    test_size=0.10,
                                                    random_state=2)

# -4- 模型训练和保存

#SVM

from sklearn.preprocessing import StandardScaler  # 归一化
# 归一化操作
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import GridSearchCV  # 在sklearn中主要是使用GridSearchCV调参

svc_model = SVC(kernel='rbf')
param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  # param_grid:我们要调参数的列表(带有参数名称作为键的字典)，此处共有14种超参数的组合来进行网格搜索，进而选择一个拟合分数最好的超平面系数。
grid_search = GridSearchCV(svc_model, param_grid, n_jobs=8, verbose=1)  # n_jobs:并行数，int类型。(-1：跟CPU核数一致；1:默认值)；verbose:日志冗长度。默认为0：不输出训练过程；1：偶尔输出；>1：对每个子模型都输出。
grid_search.fit(x_train, y_train.ravel())  # 训练，默认使用5折交叉验证
best_parameters = grid_search.best_estimator_.get_params()  # 获取最佳模型中的最佳参数
print("cv results are" % grid_search.best_params_, grid_search.cv_results_)  # grid_search.cv_results_:给出不同参数情况下的评价结果。
print("best parameters are" % grid_search.best_params_, grid_search.best_params_)  # grid_search.best_params_:已取得最佳结果的参数的组合；
print("best score are" % grid_search.best_params_, grid_search.best_score_)  # grid_search.best_score_:优化过程期间观察到的最好的评分。
# for para, val in list(best_parameters.items()):
#     print(para, val)
svm_model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])  # 最佳模型

clf = SVC()#选择svc模型
clf.fit(x_train, y_train)#模型的训练
clf.score(x_test, y_test)#模型的测试
joblib.dump(clf, 'models/SVM.pkl')
print("SVM model saved in ./models")

score = clf.score(x_test, y_test) * 100
print("SVM score is: %.3f"%score, "in test dataset")


# K近邻
knn = make_pipeline(StandardScaler(),  
                     KNeighborsClassifier(3))
knn.fit(x_train, y_train)
# 保存Model(models 文件夹要预先建立，否则会报错)
joblib.dump(knn, 'models/knn.pkl')
print("KNN model saved in ./models")

score = knn.score(x_test, y_test) * 100
print("KNN score is: %.3f"%score, "in test dataset")

# 高斯分布的贝叶斯
nbg = make_pipeline(StandardScaler(),
                    GaussianNB())
nbg.fit(x_train, y_train)

joblib.dump(nbg, 'models/GaussianNB.pkl')
print("GaussianNB model saved in ./models")

score = nbg.score(x_test, y_test) * 100
print("GaussianNB score is: %.3f"%score, "in test dataset")
# 随机森林
rfc = make_pipeline(StandardScaler(),
                    RandomForestClassifier(max_depth=6, random_state=0))
rfc.fit(x_train, y_train)

joblib.dump(rfc, 'models/RandomForest.pkl')
print("RandomForest model saved in ./models")

score = rfc.score(x_test, y_test) * 100
print("RandomForest score is: %.3f"%score,  "in test dataset")
'''
#BP神经网络
import numpy as np
def sigmoid(x):
    """
    隐含层和输出层对应的函数法则
    """
    return 1/(1+np.exp(-x))


def BP(data_tr, data_te, maxiter=600):
    # --pandas是基于numpy设计的，效率略低
    # 为提高处理效率，转换为数组
    data_tr, data_te = np.array(data_tr), np.array(data_te)

    # --隐层输入
    # -1： 代表的是隐层的阈值
    net_in = np.array([0.0, 0, -1])
    w_mid = np.random.rand(3, 4)  # 隐层权值阈值（-1x其中一个值：阈值）

    # 输出层输入
    # -1：代表输出层阈值
    out_in = np.array([0.0, 0, 0, 0, -1])
    w_out = np.random.rand(5)  # 输出层权值阈值（-1x其中一个值：阈值）
    delta_w_out = np.zeros([5])  # 存放输出层权值阈值的逆向计算误差
    delta_w_mid = np.zeros([3, 4])  # 存放因此能权值阈值的逆向计算误差
    yita = 0.1  # η： 学习速率
    Err = np.zeros([maxiter])  # 记录总体样本每迭代一次的错误率

    # 1.样本总体训练的次数
    for it in range(maxiter):

        # 衡量每一个样本的误差
        err = np.zeros([len(data_tr)])

        # 2.训练集训练一遍
        for j in range(len(data_tr)):
            net_in[:2] = data_tr[j, :2]  # 存储当前对象前两个属性值
            real = data_tr[j, 2]

            # 3.当前对象进行训练
            for i in range(4):
                out_in[i] = sigmoid(sum(net_in * w_mid[:, i]))  # 计算输出层输入
            res = sigmoid(sum(out_in * w_out))  # 获得训练结果

            err[j] = abs(real - res)

            # --先调节输出层的权值与阈值
            delta_w_out = yita * res * (1 - res) * (real - res) * out_in  # 权值调整
            delta_w_out[4] = -yita * res * (1 - res) * (real - res)  # 阈值调整
            w_out = w_out + delta_w_out

            # --隐层权值和阈值的调节
            for i in range(4):
                # 权值调整
                delta_w_mid[:, i] = yita * out_in[i] * (1 - out_in[i]) * w_out[i] * res * (1 - res) * (
                            real - res) * net_in
                # 阈值调整
                delta_w_mid[2, i] = -yita * out_in[i] * (1 - out_in[i]) * w_out[i] * res * (1 - res) * (real - res)
            w_mid = w_mid + delta_w_mid
        Err[it] = err.mean()
    plt.plot(Err)
    plt.show()

    # 存储预测误差
    err_te = np.zeros([100])

    # 预测样本100个
    for j in range(100):
        net_in[:2] = data_te[j, :2]  # 存储数据
        real = data_te[j, 2]  # 真实结果

        # net_in和w_mid的相乘过程
        for i in range(4):
            # 输入层到隐层的传输过程
            out_in[i] = sigmoid(sum(net_in * w_mid[:, i]))
        res = sigmoid(sum(out_in * w_out))  # 网络预测结果输出
        err_te[j] = abs(real - res)  # 预测误差
        #print('res:', res, ' real:', real)

    #plt.plot(err_te)
    #plt.show()


if "__main__" == __name__:
    # 1.读取样本
    data_tr = x_train
    data_te = x_test
    BP(data_tr, data_te, maxiter=600)

joblib.dump(BP, 'models/BP.pkl')
print("BP model saved in ./models")
'''