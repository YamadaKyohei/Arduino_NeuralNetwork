# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neural_network import MLPRegressor

def main():
    model = MLPRegressor(hidden_layer_sizes=(3),max_iter=10000,activation='relu',verbose=True,learning_rate_init=0.01)

    # 説明変数
    X = [[0,0],[0,1],[1,0],[1,1]]
    # 目的変数
    Y = [0,1,1,0]
    # 学習
    model.fit(X, Y)
    
    # 学習結果で推論
    pred = model.predict(X)
    print(pred)

    # 学習結果表示

    #print(model.coefs_)#これでweightが見れる！
    #print(model.intercepts_)#これでbiasが見れる！
    
    print("// ===== ここからコピペ =====")
    
    print("const float w[depth-1][MaxNode][MaxNode] = {")
    for i in range(len(model.coefs_)):
        if i is not 0:
            print(", ")
        print(np.array2string(model.coefs_[i],separator=', ').replace("[","{").replace("]","}"),end="")
    print("")
    print("};")
    print("const float bias[depth-1][MaxNode] = {")
    for i in range(len(model.intercepts_)):
        if i is not 0:
            print(", ")
        print(np.array2string(model.intercepts_[i],separator=', ').replace("[","{").replace("]","}"), end="")
    print("")
    print("};")
    
    print("// ===== ここまでコピペ =====")


if __name__ == "__main__":
    main()
