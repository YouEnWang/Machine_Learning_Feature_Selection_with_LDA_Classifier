# 目標
實做兩種特徵篩選方法(Sequential Forward Selection 和 Fisher’s Criterion)；比較 Filter-based 和 Wrapper-based 特徵篩選法的異同；並利用乳癌資料集，搭配 LDA 分類器和2-Fold CV 完成分類任務，並使用平衡分類率以評估分類器效能。
- 乳癌資料集, Breast Cancer dataset：https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

# 資料描述
1. 乳癌資料集如同Iris dataset，是在機器學習領域中常被用作演算法效能驗證的開放資料集
2. 包含兩個類別：惡性腫瘤和良性腫瘤，前者標籤為 M = malignant 而後者為 B = benign
3. 共包含569筆資料，而每筆資料皆以30種特徵(𝑁𝑠 = 30)進行描述

# 作業內容
### Part1: Sequential Forward Selection, SFS
1. 透過 LDA 和 2-Fold CV實現SFS演算法，並記錄每次iteration的 Highest validated balanced accuracy。
2. 找出最佳特徵子集合(Optimal feature subset)，並將所包含的特徵和特徵數記錄下來。

### Part2: Fisher’s Criterion
1. 實現 Fisher’s Criterion 演算法（請勿直接使用開源的 Fisher’s Criterion 套件）。
2. 計算全部 𝑁𝑠 種特徵的 Fisher’s score。
3. 根據 Step2 所計算出的結果，對特徵進行降序排列(Rank in descending order)
4. 透過 LDA 和 2-Fold CV 計算 Fisher’s score 最高 N 筆特徵(Top-N-ranked features) 之 Validated balanced accuracy。
5. 找出最佳特徵子集合(Optimal feature subset)，並將所包含的特徵和特徵數記錄下來。

# 程式執行方式
- 此次作業我依照part1跟part2設計出兩個程式，分別為main_SFS.py以及main_Fisher.py。
1. main_SFS.py
 - 透過ucimlrepo dataset直接導入Breast Cancer dataset
 - 直接執行程式，便會產生出所有結果的csv，內容包含：
  (1) 每個step的Highest validated balanced accuracy與特徵子集
  (2) Optimal feature subset的特徵數、validated balanced accuracy、依feature index做sorted的Optimal feature subset

2. main_Fisher.py
 - 透過ucimlrepo dataset直接導入Breast Cancer dataset
 - 直接執行程式，便會產生出所有結果的csv，內容包含：
  (1) 每個feature的Fisher's score(依score做降序排列)
  (2) 用最高N筆特徵做LDA的Validated balanced accuracy與特徵子集
  (3) Optimal feature subset的特徵數、validated balanced accuracy、依feature index做sorted的Optimal feature subset

# 討論
1. Sequential Forward Selection 和 Fisher’s Criterion 分別屬於 Filter-based 和 Wrapper-based 中的何種特徵篩選方法？
	- Sequential Forward Selection：Wrapper-based
  	- Fisher’s Criterion：Filter-based

2. 一般來說 Filter-based 和 Wrapper-based 各有什麼性質或優缺點？
	- Filter-based：
    		- 優點：演算法較容易實現、耗時短
   		- 缺點：沒有考慮到聯合機率、可能選擇到冗餘特徵
	- Wrapper-based：
    		- 優點：綁定分類器、考慮到聯合機率
    		- 缺點：計算複雜度大

3. 在本次作業的結果中是否有展現出跟上一題你的回答有一致的現象呢？
   - 以程式設計與執行方面，Fisher’s Criterion 的確較容易實現，且時間複雜度較低。
   - 考慮分類結果的話，SFS 的最佳特徵子集的特徵數較 Fisher’s Criterion 少。觀察 Fisher’s method 的特徵數與 Validated balanced accuracy 可以看出，很常有特徵數增加，但分類率不變的狀況發生，驗證了上一題 Fisher’s method 可能選擇到冗餘特徵的缺點。
   - 此外，SFS 的最佳特徵子集有較高的分類率，可能是因為 SFS 有考慮到聯合機率的影響。

# 心得
這次設計 SFS 演算法時，因為想要一次執行完全部的 step，所以構想出用 dictionary 逐步存儲結果的方式，讓我後續篩選特徵來進行 LDA 的時候方便很多，也比較方便觀察全部的結果。在設計 Fisher’s method 時，雖然Fisher 的數學公式跟 LDA 的數學公式很像，但我想把 Fisher 的程式設計的更符合特徵篩選演算法，因此做了一些改動，讓其可以運用於 2 個類別以上的 F-score 計算。雖然這些改動在思考的時候很花時間，但最後設計出的程式具有較大的泛用性，符合我的預期結果。
