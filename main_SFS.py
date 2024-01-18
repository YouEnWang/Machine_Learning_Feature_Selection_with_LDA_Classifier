# import the ucimlrepo dataset
from ucimlrepo import fetch_ucirepo 

'''part 1: Sequential Forward Selection'''
# import modules
import numpy as np
import math
import csv

# Split data
def split(initial_X, initial_y, change, features_subset):
    # 依照選定的feature subset來選擇X
    X_selected = []
    for i in range(len(features_subset)):
        X_selected.append(initial_X[:, features_subset[i]])

    X_selected = np.array(X_selected).T

    # 將資料分為前半(285筆)與後半(284筆)
    split_data_X = np.array_split(X_selected, 2)
    split_data_y = np.array_split(initial_y, 2)

    # 根據change更換選擇的training data和test data
    if change:                  # 前半資料為training data
        X_training, X_test = split_data_X
        y_training, y_test = split_data_y
    else:                       # 後半資料為training data
        X_test, X_training = split_data_X
        y_test, y_training = split_data_y
    
    return X_training, y_training, X_test, y_test


# calculate weight vector及bias
def get_w_b(x, y, c):
    # 分割為正負類別，將M跟B分別label為正負類別
    x_P = []
    x_N = []
    for i in range(len(x)):
        if y[i] == "M":
            x_P.append(x[i])
        else:
            x_N.append(x[i])
    
    n1 = len(x_P)                       # 正類別資料數量
    n2 = len(x_N)                       # 負類別資料數量
    # print(x_N)
    # print(x_P)

    '''計算平均向量'''
    # 正類別
    m1 = np.zeros(shape = len(x_P[0]))
    for i in range(len(m1)):
        for j in range(len(x_P)):
            m1[i] += x_P[j][i]
        
    for k in range(len(m1)):
        m1[k] = round((m1[k] / n1), 4)
    m1 = np.array([m1])
    # print(m1)

    # 負類別
    m2 = np.zeros(shape = len(x_N[0]))
    for i in range(len(m2)):
        for j in range(len(x_N)):
            m2[i] += x_N[j][i]
    
    for k in range(len(m2)):
        m2[k] = round((m2[k] / n2), 4)
    m2 = np.array([m2])
    print('m1 = ', m1)
    print('m2 = ', m2)

    '''prior probability'''
    p1 = n1/(n1 + n2)           # 正類別先驗機率
    p2 = n2/(n1 + n2)           # 負類別先驗機率

    '''covariance matrix'''
    # 正類別
    co_P = np.zeros(shape = len(m1) * len(m1))
    for i in range(len(x_P)):
        # print(((np.array(x_P[i]) - m1).T).dot(np.array(x_P[i]) - m1))
        co_P = co_P + ((x_P[i] - m1).T).dot(x_P[i] - m1)
        # print(co_P)
    covariance1 = (1/(n1 - 1)) * co_P
    # print(co_P)

    # 負類別
    co_N = np.zeros(shape = len(m2) * len(m2))
    for i in range(len(x_N)):
        co_N = co_N + ((x_N[i] - m2).T).dot(x_N[i] - m2)
    covariance2 = (1/(n2 - 1)) * co_N

    # print(covariance1)
    # print(covariance2)
    covariance = p1*covariance1 + p2*covariance2
    # print(covariance)

    '''weight vector'''
    covariance_inv = np.linalg.inv(covariance)

    w = (m1 - m2).dot(covariance_inv)
    # 取小數點下第五位作為return的值
    for i in range(len(w[0])):
        w[0][i] = round(w[0][i], 5)
    # print("w = ", w)

    '''bias'''
    b = (-1/2) * (m1 - m2).dot(covariance_inv).dot((m1 + m2).T) - math.log((c*(p1/p2)))
    # 取小數點下第五位作為return的值
    b = round(float(b[0][0]), 5)
    # print("b = ", b)

    return w, b


# Decision function
def Decision(X_test, w, b):
    # 存儲最後預測所得的label
    predict = []

    for i in range(len(X_test)):
        D = w.dot(X_test[i]) + b
        if D[0] > 0:
            predict.append("M")
        elif D[0] < 0:
            predict.append("B")
        else:
            predict.append("false")
    
    return predict


# 計算分類率
def classification_rate(y_test, predict):
    # 預測正確的資料總數
    True_prediction = 0

    # 將predict的label與test data的label做比對
    for i in range(len(predict)):
        if predict[i] == y_test[i]:
            True_prediction += 1
    
    # 分類率
    # print(True_prediction)
    CR = round(True_prediction / len(y_test), 5) * 100
    return CR


# LDA Algorithm
def LDA(X_training, y_training, X_test, y_test, c):
    # Decision function
    w, b = get_w_b(X_training, y_training, c)

    # Prediction
    predict = Decision(X_test, w, b)
    CR = classification_rate(y_test, predict)

    print("predict = ", predict)
    print("CR = ", CR)

    return CR


# Sequential Forward Selection
def SFS(X, y, result_dict):
    '''LDA classification'''
    initial_features = [i for i in range(30)]
    change = [0, 1]         # 做2-fold CV
    c = 1                   # 正負類別懲罰權重比
    
    remaining_features = initial_features.copy()
    selected_features = []

    # 總共有Ns次的step
    Ns = 30
    for i in range(Ns):
        # 將上一次iteration時選到的features從remaining_features中刪掉
        # np.setdiff1d()會return兩個array中的未重複項
        remaining_features = np.setdiff1d(remaining_features, selected_features)
        print("remaining features = ", remaining_features)

        # 開始SFS當次的iteration
        for j in range(len(remaining_features)):
            print("selected_features =", selected_features)
            # 用copy()避免更動到selected_features的值
            subset = selected_features.copy()
            # print("subset =", subset)
            subset.append(remaining_features[j])
            print("subset =", subset)
            result_dict["features_subset"][i].append(subset)
            for k in change:
                # split data
                X_training, y_training, X_test, y_test = split(X, y, change[k], subset)
                print("X_training = ", X_training)

                CR = LDA(X_training, y_training, X_test, y_test, c)
                
                if k == 0:
                    result_dict["first_half_CR"][i].append(CR)
                else:
                    result_dict["second_half_CR"][i].append(CR)
            
            balanced_CR = round(((result_dict["first_half_CR"][i][j] + result_dict["second_half_CR"][i][j]) / 2), 2)
            result_dict["balanced_CR"][i].append(balanced_CR)
        
        # Highest validated balanced accuracy
        max_CR = max(result_dict["balanced_CR"][i])
        max_CR_index = result_dict["balanced_CR"][i].index(max_CR)
        
        result_dict["Highest_CR"].append(max_CR)
        result_dict["best_subset"].append(result_dict["features_subset"][i][max_CR_index])

        # 更新selected features，已進行下一次的篩選
        selected_features = result_dict["best_subset"][i]

        # 將result_dict新增空間，用來儲存下一次篩選的結果
        result_dict["features_subset"].append([])
        result_dict["first_half_CR"].append([])
        result_dict["second_half_CR"].append([])
        result_dict["balanced_CR"].append([])
        

# main function
def main():
    '''load the ucimlrepo dataset'''
    # fetch dataset 
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
    
    # data (as pandas dataframes) 
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    # print(X.head())

    X = np.array(X)
    # print(X)
    y = np.asarray(y).T
    y = y[0]
    # print(y)
    
    get_w_b(X, y, 1)
    # SFS
    result_dict = {"features_subset": [[]], 
                   "first_half_CR": [[]], 
                   "second_half_CR": [[]], 
                   "balanced_CR": [[]], 
                   "best_subset": [], 
                   "Highest_CR": []}        # 儲存SFS的所有結果
    
    SFS(X, y, result_dict)
    
    print(result_dict["Highest_CR"])
    print(result_dict["best_subset"])
    
    # 最佳特徵子集合
    Highest_Highest_CR = max(result_dict["Highest_CR"])
    index = result_dict["Highest_CR"].index(Highest_Highest_CR)
    Optimal_feature = result_dict["best_subset"][index]
    print("Part1: Sequential Forward Selection, SFS")
    print("最佳特徵子集之CR =", Highest_Highest_CR)
    print("最佳特徵子集特徵數 =", len(Optimal_feature))
    print("最佳特徵子集 =", sorted(Optimal_feature))
    
    # 將結果輸出
    title1 = ["Highest validated balanced accuracy", "feature subset"]
    title2 = ["特徵數", "validated balanced accuracy", "Optimal feature subset"]
    with open("SFS_result.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(title1)

        dict_writer = csv.DictWriter(file, fieldnames=["Highest_CR", "best_subset"])
        for i in range(30):
            dict_writer.writerow({"Highest_CR": result_dict['Highest_CR'][i], "best_subset": result_dict['best_subset'][i]})
        # for i in range(30):
        #     file.write(f"{result_dict['Highest_CR'][i]}, {result_dict['best_subset'][i]}\n")

        # 最佳特徵子集合
        file.write("\n\nOptimal feature subset\n")
        writer.writerow(title2)
        writer.writerow([len(Optimal_feature), Highest_Highest_CR, sorted(Optimal_feature)])

    
    file.close()

if __name__ == "__main__":
    main()