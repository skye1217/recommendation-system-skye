#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 协同过滤-基于SVD
# dataset: /Users/yangguangqiang/Downloads/ml-latest-small/ratings.csv
# info: userId	movieId	rating	timestamp
# 结果： User矩阵(num_user,num_LF) Item矩阵:(num_LF,num_i)
import pandas as pd
import numpy as np

class LFM():
    def __init__(self, alpha, reg_user, reg_item,num_LF=10,num_ep=10, columns=["userId", "movieId", "rating"]):
       self.alpha = alpha  # 学习率
       self.reg_user = reg_user    # User矩阵的正则系数
       self.reg_item = reg_item   # item矩阵的正则系数
       self.num_LF = num_LF  # 隐式类别数量:结果矩阵的维度
       self.num_ep = num_ep    # 最大迭代次数
       self.columns = columns #输入包含的信息
    def fit(self,Data):
        self.data = pd.DataFrame(Data)

        self.users_ratings = Data.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = Data.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        self.globalMean = self.data[self.columns[2]].mean()

        self.User, self.Item = self.sgd()
        
    def _init_matrix(self): # 初始化 User和item矩阵，0，1之间的随机值作为初始值

        # User-LF
        User= dict(zip(self.users_ratings.index,np.random.rand(len(self.users_ratings), 
            self.num_LF).astype(np.float32)))
        # Item-LF
        Item = dict(zip(self.items_ratings.index,np.random.rand(len(self.items_ratings),
                    self.num_LF).astype(np.float32)))
        return User, Item
    def sgd(self): #随机梯度下降

        P, Q = self._init_matrix()

        for i in range(self.num_ep): #迭代
            print("iter%d"%i)
            error_list = []
            for uid, iid, r_ui in self.data.itertuples(index=False):
                # User-LF P
                ## Item-LF Q
                v_pu = P[uid] #用户向量
                v_qi = Q[iid] #物品向量
                err = np.float32(r_ui - np.dot(v_pu, v_qi))
                #套公式
                v_pu += self.alpha * (err * v_qi - self.reg_user * v_pu)
                v_qi += self.alpha * (err * v_pu - self.reg_item * v_qi)

                P[uid] = v_pu 
                Q[iid] = v_qi

                error_list.append(err ** 2)
            print('train error:',np.sqrt(np.mean(error_list)))
        return P, Q
    def predict(self, uid, iid):
        # 如果uid或iid不在，我们使用全剧平均分作为预测结果返回
        if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
            return self.globalMean

        p_u = self.User[uid]
        q_i = self.Item[iid]

        return np.dot(p_u, q_i) #预测分数，点乘

if __name__ == '__main__':
    dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
    dataset = pd.read_csv("/Users/yangguangqiang/Music/career-2021/recommend system/ml-latest-small/ratings.csv", usecols=range(3), dtype=dict(dtype))

    model= LFM(0.02, 0.01, 0.01, 10, 20, ["userId", "movieId", "rating"])
    model.fit(dataset)

    while True:
        uid = input("input User id: ") 
        iid = input("input Movie id: ")
        print('predicted rating: ',model.predict(int(uid), int(iid)))

        














