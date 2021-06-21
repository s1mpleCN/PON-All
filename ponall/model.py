import joblib
import math
import copy
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import os
import shutil
from . import feature_extraction, config

# import logging
logger = logging.getLogger(__name__)


# 根据cv训练集计算所有的LR值
def calculate_LR(df1, df2):
    """
    df1:cv training set
    df2:cv test set
    """
    # log ((2+c)/(1+c)) + log ((2+c)/ (1+c)), {c==1}

    # 有害和中性注释的字典
    p = {}
    n = {}
    for index, row in df1.iterrows():
        if (pd.isna(row['ancestor'])):
            continue
        for i in row['ancestor'].split(','):
            if i not in p.keys():
                p[i] = 1
                n[i] = 1
            if (row['is_del'] == 1):
                p[i] += 1
            else:
                n[i] += 1
    l = copy.deepcopy(p)
    for i in l.keys():
        l[i] = math.log(p[i] / n[i])
    l

    # 求和计算每个蛋白的lr
    def LR_add(x):
        sum = 0
        if (pd.isna(x)):
            return sum
        for i in x.split(','):
            if i in l:
                sum = sum + l[i]
        return sum

    df1['LR'] = df1['ancestor'].apply(lambda x: LR_add(x))
    df2['LR'] = df2['ancestor'].apply(lambda x: LR_add(x))
    df1 = df1.drop(columns=['ancestor'])
    df2 = df2.drop(columns=['ancestor'])
    return df1, df2


# 根据cv训练集计算所有的LR值
def calculate_PA(df1, df2):
    """
    df1:cv training set
    df2:cv test set
    """
    # log ((2+c)/(1+c)) + log ((2+c)/ (1+c)), {c==1}

    # 有害和中性注释的字典
    p = {}
    n = {}
    for index, row in df1.iterrows():
        if (pd.isna(row['site'])):
            continue
        for i in row['site'].split(','):
            if i != '':
                if i not in p.keys():
                    p[i] = 1
                    n[i] = 1
                if (row['is_del'] == 1):
                    p[i] += 1
                else:
                    n[i] += 1

    s = copy.deepcopy(p)
    for i in s.keys():
        s[i] = math.log(p[i] / n[i])
    s

    # 求和计算每个蛋白的pa
    def PA_add(x):
        sum = 0
        if (pd.isna(x)):
            return sum
        for i in x.split(','):
            if i != '' and i in s:
                sum = sum + s[i]
        return sum

    df1['PA'] = df1['site'].apply(lambda x: PA_add(x))
    df2['PA'] = df2['site'].apply(lambda x: PA_add(x))
    df1 = df1.drop(columns=['site'])
    df2 = df2.drop(columns=['site'])
    return df1, df2


class PonAll:
    def __init__(self):
        """
        model
        self.Estimator: 使用的训练器模型
        self.kwargs: 模型参数
        """
        self.model_path = config.model_path
        self.model_path_N = config.model_path_N
        self.model = joblib.load(self.model_path)
        self.model_N = joblib.load(self.model_path_N)

    def check_X(self, X):
        # 检查类型
        if not isinstance(X, pd.DataFrame):
            raise RuntimeError("The input is not the object of pandas.DataFrame")
        # 检查特征
        all_features = set(self.model.to_list() + self.model_N.to_list())
        input_data_features = set(X.columns.to_list())
        reduce_features = all_features - input_data_features
        if len(reduce_features) > 0:
            raise RuntimeError("缺少特征:%s" % reduce_features)
        return True

    def predict(self, n, seq, aa, kind, ip):
        """
        预测
        :param seq: 氨基酸序列，不包含名称
        :param aa: 变异，索引从1开始，e.g. A1B
        :return: 预测结果
        """
        # print('开始预测')
        # err_list,df2 = feature_extraction.get_all_features(n, seq, aa, kind)
        df2 = feature_extraction.get_all_features(n, seq, aa, kind)
        err_list = df2[df2['msg'] != '']
        # print('错误')
        # print(err_list)
        df2 = df2[df2['msg'] == '']
        # print('特征')
        # print(df2)
        df2.to_csv('{}_test.csv'.format(ip), index=None)
        err_list['confidence'] = ''
        err_list['pred'] = ''
        err_list = err_list[['id', 'confidence', 'pred', 'msg']]
        confidence, pred = self._predict(df2.iloc[:, 2:], kind, ip)
        confidence = pd.DataFrame(confidence)
        confidence.rename(columns={0: 'confidence'}, inplace=True)
        pred = pd.DataFrame(pred)
        pred.rename(columns={0: 'pred'}, inplace=True)
        pred = pd.merge(confidence, pred, left_index=True, right_index=True, how='inner')
        pred = pd.merge(df2, pred, left_index=True, right_index=True, how='inner')
        pred = pred[['id', 'confidence', 'pred', 'msg']]
        pred = pd.concat([err_list, pred], axis=0)
        pred.set_index(["id"], inplace=True)
        # print('结果')
        # print(pred)
        return pred

    def _predict(self, df2, kind, ip):
        # 检查 X
        # print('开始模型预测！！！！')
        # 创建专属ip的文件夹
        dirs = config.test_bootstrap_path + '{}/'.format(ip)
        # 所有特征值
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        # self.check_X(df2)
        # file_train = config.all_bootstrap_path

        df1 = pd.read_csv(config.all_path)
        file_train = dirs
        # print('11111111111111111')
        if (kind in ['uniprot id', 'ensembl id', 'vcf']):
            # print('22222222222')
            # file_train = 'F:/PONP3-2021-1-17PDF美化版/Protein/'
            # file_train = 'F:/PONP3-2021-1-17PDF美化版/Uniprot/'
            # file_train = 'F:/PONP3-2021-1-17PDF美化版/Ensemblid/'
            # file_train = 'F:/PONP3-2021-1-17PDF美化版/Vcf/'

            rfe = joblib.load(config.model_path)
            # sample
            df1, df2 = calculate_LR(df1, df2)
            df1, df2 = calculate_PA(df1, df2)
            X_test = df2
            for i in range(200):
                train = df1.sample(frac=1.0, replace=True)
                train.to_csv(file_train + "bootstrap_Combine_train_{}.csv".format(i), float_format='%.3f', index=None)

            # 200bootrstrap

            for i in range(200):
                data1 = pd.read_csv(file_train + "bootstrap_Combine_train_{}.csv".format(i))
                y_train = data1.is_del.values
                X_train = data1.iloc[:, data1.columns != "nutation"].iloc[:, 5:]

                model = lgb.LGBMClassifier()
                model.fit(pd.DataFrame(rfe.transform(X_train)), y_train)
                p_test = model.predict_proba(pd.DataFrame(rfe.transform(X_test)).astype('float'))
                df = pd.DataFrame(p_test[:, -1], df2.index)
                df.to_csv(file_train + "bootstrap_Combine_{}_re.csv".format(i))

            data1 = pd.read_csv(file_train + "bootstrap_Combine_0_re.csv")
            for i in range(1, 200):
                data2 = pd.read_csv(file_train + "bootstrap_Combine_{}_re.csv".format(i))
                data1 = pd.concat([data1, data2], ignore_index=True)
            # ##平均值和标准差
            d_mean = data1.groupby('Unnamed: 0')['0'].mean()
            d_std = data1.groupby('Unnamed: 0')['0'].std()
            # ###连接方式还需要探讨
            data1 = pd.merge(pd.DataFrame(d_mean), pd.DataFrame(d_std), on='Unnamed: 0', how='outer')
            # ##置信水平
            k = 20 ** 0.5

            def getA(row):
                return row['0_x'] - k * row['0_y']

            def getB(row):
                return row['0_x'] + k * row['0_y']

            data1["A"] = data1.apply(lambda row: getA(row), axis=1)
            data1["B"] = data1.apply(lambda row: getB(row), axis=1)

            # ##删除不确定项
            def getT(row):
                if (row['A'] < 0.5) and (row['B'] > 0.5):
                    return 1
                return 0

            data1["T"] = data1.apply(lambda row: getT(row), axis=1)
            data2 = pd.merge(data1[['T']], df2, left_index=True, right_index=True, how='outer')
            # data3 = data2[data2['T'] == 0]
            # # ##可以换成drop函数
            # del data3['T']
            # bootstrap完训练
            y_train = df1.is_del.values
            X_train = df1.iloc[:, df1.columns != "nutation"].iloc[:, 5:]
            # 筛选完的验证集
            X_test = data2
            model = lgb.LGBMClassifier()
            model.fit(pd.DataFrame(rfe.transform(X_train)), y_train)
            confidence = model.predict_proba(pd.DataFrame(rfe.transform(X_test.iloc[:, 1:])).astype('float'))
            y_pred = model.predict(pd.DataFrame(rfe.transform(X_test.iloc[:, 1:])).astype('float'))
            col = data2['T']
            for index in col.index:
                # 如果T==1，判断unknown
                if col[index] == 1:
                    confidence[index][0] = 0
                    confidence[index][1] = 0
                    y_pred[index] = -1
        else:
            # print('33333333')
            # file_train = 'F:/PONP3-2021-1-17PDF美化版/Seq/'
            # file_train = 'F:/PONP3-2021-1-17PDF美化版/Entrezid/'
            rfe = joblib.load(config.model_path_N)
            # sample
            X_test = df2
            del df1['ancestor']
            df1 = df1.iloc[:, df1.columns != "nutation"].iloc[:, 4:-1]
            for i in range(200):
                train = df1.sample(frac=1.0, replace=True)
                train.to_csv(file_train + "bootstrap_Combine_train_{}.csv".format(i), float_format='%.3f', index=None)

            # # 200bootrstrap
            #
            for i in range(200):
                data1 = pd.read_csv(file_train + "bootstrap_Combine_train_{}.csv".format(i))
                y_train = data1.is_del.values
                X_train = data1.iloc[:, 1:]
                model = lgb.LGBMClassifier()
                model.fit(pd.DataFrame(rfe.transform(X_train)), y_train)
                p_test = model.predict_proba(pd.DataFrame(rfe.transform(X_test)).astype('float'))
                # p_test = model.predict_proba(pd.DataFrame(rfe.transform(X_test)))
                df = pd.DataFrame(p_test[:, -1], df2.index)
                df.to_csv(file_train + "bootstrap_Combine_{}_re.csv".format(i))
            data1 = pd.read_csv(file_train + "bootstrap_Combine_0_re.csv")
            for i in range(1, 200):
                data2 = pd.read_csv(file_train + "bootstrap_Combine_{}_re.csv".format(i))
                data1 = pd.concat([data1, data2], ignore_index=True)
            # ##平均值和标准差
            d_mean = data1.groupby('Unnamed: 0')['0'].mean()
            d_std = data1.groupby('Unnamed: 0')['0'].std()
            # ###连接方式还需要探讨
            data1 = pd.merge(pd.DataFrame(d_mean), pd.DataFrame(d_std), on='Unnamed: 0', how='outer')
            # ##置信水平
            k = 20 ** 0.5

            def getA(row):
                return row['0_x'] - k * row['0_y']

            def getB(row):
                return row['0_x'] + k * row['0_y']

            data1["A"] = data1.apply(lambda row: getA(row), axis=1)
            data1["B"] = data1.apply(lambda row: getB(row), axis=1)

            # ##删除不确定项
            def getT(row):
                if (row['A'] < 0.5) and (row['B'] > 0.5):
                    return 1
                return 0

            data1["T"] = data1.apply(lambda row: getT(row), axis=1)
            data2 = pd.merge(data1[['T']], df2, left_index=True, right_index=True, how='outer')
            # data3 = data2[data2['T'] == 0]
            # # ##可以换成drop函数
            # del data3['T']
            # bootstrap完训练
            y_train = df1.is_del.values
            X_train = df1.iloc[:, 1:]
            # 筛选完的验证集
            X_test = data2
            model = lgb.LGBMClassifier()
            model.fit(pd.DataFrame(rfe.transform(X_train)), y_train)
            confidence = model.predict_proba(pd.DataFrame(rfe.transform(X_test.iloc[:, 1:])).astype('float'))
            y_pred = model.predict(pd.DataFrame(rfe.transform(X_test.iloc[:, 1:])).astype('float'))
            col = data2['T']
            for index in col.index:
                # 如果T==1，判断unknown
                if col[index] == 1:
                    confidence[index][0] = 0
                    confidence[index][1] = 0
                    y_pred[index] = -1
        # 删除整个文件夹
        shutil.rmtree(file_train)
        return np.round(confidence[:, -1], 3), y_pred
