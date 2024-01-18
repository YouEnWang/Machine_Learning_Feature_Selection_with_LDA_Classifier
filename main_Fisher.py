# import the ucimlrepo dataset
from ucimlrepo import fetch_ucirepo 

'''part 2: Fisher’s Criterion'''
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


# Within-class scatter matrix, Between-class scatter matrix
def Scatter_Matrix(x, y):
    # 所有的class
    total_class = ["M", "B"]
    
    # 依照class分割資料
    x_class = []        # 依照class分割之資料的訓練資料子集
    for i in range(len(total_class)):
        x_class.append([])
        for j in range(len(x)):
            if y[j] == total_class[i]:
                x_class[i].append(x[j])
    # x_class = [x_M[], x_B[]]

    # 各個訓練資料子集的資料數量
    n_list = []
    for i in range(len(x_class)):
        n_list.append(len(x_class[i]))
    # n_list = [n_M, n_B]
    
    n_total = 0
    for i in range(len(n_list)):
        n_total += n_list[i]

    '''計算平均向量'''
    # 各個class的平均向量
    m_list = []
    # overall mean
    m_total = np.zeros(shape = len(x_class[0][0]))
    
    for i in range(len(total_class)):
        # 乳癌資料集: x_P = x_class[0], x_N = x_class[1]
        # 乳癌資料集: m_P = m_list[0], m_N = m_list[1]
        m_list.append(np.zeros(shape = len(x_class[i][0])))
        for j in range(n_list[i]):
            m_list[i] += x_class[i][j]
        
        # overall mean
        m_total = m_total + m_list[i]
        
        # 對第i個m做四捨五入
        m_list[i] = np.round(((1/n_list[i]) * m_list[i] ), 6)
        m_list[i] = np.array([m_list[i]])
    print('m1 = ', m_list[0])
    print('m2 = ', m_list[1])
    
    # overall mean
    m_total = (1/n_total) * m_total
    # print('m_total = ', m_total)

    '''prior probability'''
    # 各個class的先驗機率
    p_list = []
    for i in range(len(n_list)):
        p_list.append((n_list[i] / n_total))
    
    '''Within-class scatter matrix'''
    # 各個class的Within-class scatter
    Sw_list = []
    Sw = np.zeros(shape = len(m_list[0]) * len(m_list[0]))
    for i in range(len(total_class)):
        Sw_list.append(np.zeros(shape = len(m_list[i]) * len(m_list[i])))
        for j in range(n_list[i]):
            Sw_list[i] = Sw_list[i] + ((x_class[i][j] - m_list[i]).T).dot(x_class[i][j] - m_list[i])
            
        Sw_list[i] = p_list[i] * (1/n_list[i]) * Sw_list[i]

        # Within-class scatter matrix
        Sw = Sw + Sw_list[i]
    
    # print("x_class[0][0] =", x_class[0][0])
    # print("Sw_list[0] =", Sw_list[0].shape, len(Sw_list))

    '''Between-class scatter matrix'''
    # 各個class的Between-class scatter
    Sb_list = []
    Sb = np.zeros(shape = len(m_list[0]) * len(m_list[0]))
    for i in range(len(total_class)):
        Sb_list.append(np.zeros(shape = len(m_list[i]) * len(m_list[i])))
        Sb_list[i] = n_list[i] * (((m_list[i] - m_total).T).dot(m_list[i] - m_total))

        # Between-class scatter matrix
        Sb = Sb + Sb_list[i]

    return Sb, Sw


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


# Fisher's method
def Fisher(X, y, result_dict):
    '''Fisher's score'''
    Sb, Sw = Scatter_Matrix(X, y)
    # 紀錄feature index與該feature的F-score
    F_score_list = []
    for i in range(len(X[0])):
        F_score_list.append([i, round((Sb[i][i] / Sw[i][i]), 6)])
    
    # 依照F-score做降序排列
    F_score_list = sorted(F_score_list, key=lambda x: x[1], reverse=True)
    print("F_score_list =", F_score_list)

    '''LDA classification'''
    change = [0, 1]         # 做2-fold CV
    c = 1                   # 正負類別懲罰權重比
    initial_features = []
    for i in range(len(F_score_list)):
        initial_features.append(F_score_list[i][0])
    
    # 總共有Ns次的step
    Ns = 30
    for i in range(Ns):
        # 將initial_features加入selected_features
        subset = initial_features[0:i+1]
        print("subset =", subset)
        result_dict["features_subset"].append(subset)

        for k in change:
            # split data
            X_training, y_training, X_test, y_test = split(X, y, change[k], subset)
            # print("X_training = ", X_training)

            CR = LDA(X_training, y_training, X_test, y_test, c)
                
            if k == 0:
                result_dict["first_half_CR"].append(CR)
            else:
                result_dict["second_half_CR"].append(CR)
        
        balanced_CR = round(((result_dict["first_half_CR"][i] + result_dict["second_half_CR"][i]) / 2), 2)
        result_dict["balanced_CR"].append(balanced_CR)
    
    return F_score_list

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

    # Fisher's method
    result_dict = {"features_subset": [], 
                   "first_half_CR": [], 
                   "second_half_CR": [], 
                   "balanced_CR": []}        # 儲存Fisher's method的所有結果
    F_score_list = Fisher(X, y, result_dict)

    # 最佳特徵子集合
    Highest_CR = max(result_dict["balanced_CR"])
    index = result_dict["balanced_CR"].index(Highest_CR)
    Optimal_feature = result_dict["features_subset"][index]
    print("Part2: Fisher’s Criterion")
    print("最佳特徵子集之CR =", Highest_CR)
    print("最佳特徵子集特徵數 =", len(Optimal_feature))
    print("最佳特徵子集 =", sorted(Optimal_feature))

    # 將結果輸出
    title1 = ["Validated balanced accuracy", "feature subset"]
    title2 = ["特徵數", "validated balanced accuracy", "Optimal feature subset"]
    with open("Fisher_result.csv", "a", newline="") as file:
        file.write("feature index,")
        for i in range(len(F_score_list)):
            file.write(f"{F_score_list[i][0]},")
        file.write("\nFisher\'s score,")
        for i in range(len(F_score_list)):
            file.write(f"{'%.3f'%round(F_score_list[i][1], 3)},")
        file.write("\n\n")

        writer = csv.writer(file)
        writer.writerow(title1)

        dict_writer = csv.DictWriter(file, fieldnames=["balanced_CR", "features_subset"])
        for i in range(30):
            dict_writer.writerow({"balanced_CR": result_dict['balanced_CR'][i], "features_subset": result_dict['features_subset'][i]})

        # 最佳特徵子集合
        file.write("\n\nOptimal feature subset\n")
        writer.writerow(title2)
        writer.writerow([len(Optimal_feature), Highest_CR, sorted(Optimal_feature)])

    
    file.close()
    

if __name__ == "__main__":
    main()