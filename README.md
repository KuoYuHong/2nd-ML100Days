# 2nd-ML100Days


### 機器學習100天挑戰賽-記錄一些個人的心得和紀錄

```
近年來，AI 應用已無所不在，不論在新創或是傳產領域，都可能透過機器學習解決過去難以解決的問題。
但目前台灣企業在 AI 導入的腳步仍然緩慢，除了人才嚴重短缺，教育資源無法即時跟上產業變異也是原因之一。
因此，我們發起了「 機器學習 百日馬拉松 」教練陪跑計劃，翻轉傳統上課模式，以自主練習為主，
幫助你獲得最大學習成效，搶先一步進入 AI 人工智能領域。
```

![ML100Days](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E7%99%BE%E6%97%A5%E9%A6%AC%E6%8B%89%E6%9D%BE%E5%AD%B8%E7%BF%92%E8%B7%AF%E7%B7%9A.png)

## 遊戲規則

* 每天官網上會公布題目，下載後**根據題目的要求來寫作業**，搭配官網PDF教學、補充連結輔助解題，也有專屬論壇上有陪跑教練可問答
* 每次更新題目三到四題上去，可累積起來做或是超前做，不過還是盡量一天做一題為主，**養成每天練習的習慣**
* 提交之後不會有人批改作業，會有**範例解答可供參考**，檢驗自己的學習成果
* 日後會有**期中考、期末考**，內容為**五天的Kaggle競賽**，盡力取得更好的名次！
* 提交**完整100天作業**，兩次**Kaggle競賽完賽**，才算是**完成本次馬拉松挑戰！**

### :star:為個人實作起來認為的難易度，最多:star::star::star::star::star:

### :point_right: 我的[Kaggle](https://www.kaggle.com/kuoyuhong)

## 主題一：資料清理數據前處理

### **Day_001_HW** (2019-04-16) － 資料介紹與評估資料：
難易度：:star:

申論題主要讓我們熟悉**找資料**、**觀察資料**，面對資料時要**思考的問題**、**如何解決**？
程式題練習簡單的MSE函式計算誤差，日後會運用到**許多數學公式與統計**，要多熟悉

**補充：**<br>
常見於迴歸問題的評估指標

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

常見於分類問題的指標
* Binary Cross Entropy (CE)

---

### **Day_002_HW** (2019-04-17) － EDA-1/讀取資料EDA: Data summary：
難易度：:star:

這次作業主要用到[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)的資料集

作業主要練習資料的row和col、欄位、顯示資料等等的熟悉

---

### **Day_003_HW** (2019-04-18) － 3-1如何新建一個 dataframe?3-2 如何讀取其他資料? (非 csv 的資料)：
難易度：:star::star:

練習**DataFrame的操作**以及檔案的存儲，日後會有非常多機會操作Pandas的DF

[Pickle檔案](https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/373209/
)
：實現python物件的**序列化和反序列化**

讀取圖片的話以前曾經用過**PIL套件**，使用提供的資料經過資料處理後弄一些圖片出來，難易算是普通

---

### **Day_004_HW** (2019-04-19) － EDA: 欄位的資料類型介紹及處理：

難易度：:star::star:

處理**類別型的資料**，也接觸到**Label Encoder vs. One Hot Encoder**：
Label encoding: 把每個類別 mapping 到某個整數，不會增加新欄位
One Hot encoding: 為每個類別新增一個欄位，用 0/1 表示是否

實用連接：<br>
[Label Encoder vs. One Hot Encoder in Machine Learning](https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
)<br>
[機器學習中的 Label Encoder和One Hot Encoder](https://kknews.cc/zh-tw/other/kba3lvv.html)

第一次接觸花了一些時間理解

---

### **Day_005_HW** (2019-04-20) － EDA資料分佈：

難易度：:star::star:

針對自己有興趣的欄位觀察資料分布、畫出直方圖，在畫圖花了較多時間操作和理解，有許多客製化方法決定圖的樣子
```
bins: 直方圖的柱數，默認為10
edgecolor: 直方圖邊框顏色
alpha: 透明度
figsize：圖表長寬
```
實用連結：
[Python筆記—matplotlib 創建圖例](https://zhuanlan.zhihu.com/p/37406730)

---

### **Day_006_HW** (2019-04-21) － EDA: Outlier 及處理：

難易度：:star::star::star::star:

觀察資料當中可能有outlier的欄位、解釋可能的原因，把篩選欄位印出圖表，
在理解ECDF(Emprical Cumulative Density Plot)的地方較久，對於檢查/處理異常值的操作要再熟悉些

實用連結：<br>
[Ways to Detect and Remove the Outliers](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba)<br>
[ECDF with Python](https://stackoverflow.com/questions/14006520/ecdf-in-python-without-step-function)

---

### **Day_007_HW** (2019-04-22) － 常用的數值取代：中位數與分位數連續數值標準化：

難易度：:star::star::star:

對NA值用不同方法進行填補，以及數值標準化

實用連結：<br>
[Is it a good practice to always scale/normalize data for machine learning?](https://stats.stackexchange.com/questions/189652/is-it-a-good-practice-to-always-scale-normalize-data-for-machine-learning)

---

### **Day_008_HW** (2019-04-23) － DataFrame operationData frame merge/常用的 DataFrame 操作：

難易度：:star::star::star:

使用pd.cut方法將資料分組，並畫出箱型圖觀察
```
pd.cut()等寬劃分 #每一組組距一樣<br>
pd.qcut()等頻劃分 #每一組會出現的頻率一樣
```
實用連結：<br>
[pandas的cut&qcut函數](https://medium.com/@morris_tai/pandas%E7%9A%84cut-qcut%E5%87%BD%E6%95%B8-93c244e34cfc)

---

### **Day_009_HW** (2019-04-24) － 程式實作 EDA: correlation/相關係數簡介：

難易度：:star:

熟悉相關係數，以randint和normal方式隨機產生數值畫出scatter plot圖表

```
np.random.randint(low, high=None, size=None, dtype='l') #返回隨機整數，範圍區間為[low,high），包含low，不包含high，size為數組維度大小，

np.random.normal()：
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
```
實用連結：<br>
[Guess the Correlation 相對係數小遊戲](http://guessthecorrelation.com/)：考驗自己對相關係數的敏感度

---

### **Day_010_HW** (2019-04-25) － EDA from Correlation：

難易度：:star::star::star:

當使用的圖表看不出規律或如何解讀時，使用不同的方式呈現圖表<br>
找出自己想要的欄位進行資料處理時花費較多時間，摸索了一下
```
.unique()函數：在list中只取不重複的物件，再由小到大排列

scikit-learn中fit_transform()與transform()區別：
二者的功能都是對資料進行某種統一處理（比如標準化~N(0,1)，將資料縮放(映射)到某個固定區間，歸一化，正則化等）

boxplot(x, rot=45, fontsize=15)
在X軸旋轉標籤角度，調整字體大小

quantile函數：
quantile(q) #0 <= q <= 1，只限於pandas的DataFrame使用，Series無法
```
實用連結：<br>
[Python 中用matplotlib 繪製盒狀圖（Boxplots）和小提琴圖（Violinplots）](http://blog.topspeedsnail.com/archives/737)<br>
[scikit-learn數據預處理fit_transform()與transform()的區別](https://blog.csdn.net/anecdotegyb/article/details/74857055)<br>
[p分位函數（四分位數）概念與pandas中的quantile函數](https://blog.csdn.net/u011327333/article/details/71263081)

---

### **Day_011_HW** (2019-04-26) － EDA: 不同數值範圍間的特徵如何檢視/繪圖與樣式Kernel Density Estimation (KDE)：

難易度：:star::star::star:

針對年齡分組與排序，畫出KDE圖和長條圖<br>
在使用seaborn套件畫KDE圖時花較多時間摸索

```
label = 'target == 1'：在圖表中顯示說明的圖例

Seaborn方法：(matplotlib基礎)
對於長條圖而言，Seaborn有 distplot() 方法，可以將單變數分佈的長條圖和kde同時繪製出來
```

實用連結：<br>
[Python Graph Gallery](https://python-graph-gallery.com/)：<br>
整合了許多Python繪圖函數寫法，可當成查詢用的工具手冊

[R Graph Gallery](https://www.r-graph-gallery.com/)：<br>
整合了許多R繪圖函數寫法，可當成查詢用的工具手冊

[python3.x-seaborn.heatmap隨筆](https://zhuanlan.zhihu.com/p/35494575)

---

### **Day_012_HW** (2019-04-27) － EDA: 把連續型變數離散化：

難易度：:star:

熟悉數值的離散化的調整工具<br>
複習pd.cut()函數，這在作業中出現過滿多次了

```
pd.cut()等寬劃分 #每一組組距一樣
pd.qcut()等頻劃分 #每一組會出現的頻率一樣

分組當中'(' 表示不包含, ']' 表示包含，EX：(0, 10], (10, 20] → 1~10、11~20
```

實用連結：<br>
[連續特徵的離散化 : 在什麼情況下可以獲得更好的效果(知乎)](https://www.zhihu.com/question/31989952)<br>
離散化的理由：儲存空間小，計算快，降低異常干擾與過擬合(ovefitting)的風險

[pandas的cut&qcut函數](https://medium.com/@morris_tai/pandas%E7%9A%84cut-qcut%E5%87%BD%E6%95%B8-93c244e34cfc)

---

### **Day_013_HW** (2019-04-28) － 程式實作 把連續型變數離散化：

難易度：:star::star::star::star:

離散化自己有興趣的欄位，有些欄位比較難弄，弄了滿多時間，參數一直調不好:joy:<br>
需要參考到前面Day11、12的方法，也算是在複習前面的部分<br>
有許多觀念會打結在一起，需要能夠融會貫通才行

實用連結：<br>
[seaborn入門（二）：barplot與countplot](https://zhuanlan.zhihu.com/p/24553277)

---

### **Day_014_HW** (2019-04-29) － Subplots：

難易度：:star::star:

使用subplot在不同位置畫圖形出來，同樣也是要觀察資料、清理NA值<br>
有許多好用的畫圖方法，需要來好好研究

實用連結：<br>
[Subplot](https://matplotlib.org/examples/pylab_examples/subplots_demo.html)<br>
[Seaborn.jointplot](https://seaborn.pydata.org/generated/seaborn.jointplot.html)

---

### **Day_015_HW** (2019-04-30) － Heatmap & Grid-plot：

難易度：:star::star:

練習使用random函數，有滿多種沒用過的方法<br>
之前曾經練習過subplot也複習了一下，一些新的畫圖方法也要熟記
```
numpy.random.uniform(low,high,size) #最小值、最大值，size可為[a,b]，A乘B的矩陣

np.random.normal()：
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
vmin, vmax : 顯示的數據值的最大和最小的範圍
cmap : matplotlib顏色表名稱或對象，或顏色列表，可選從數據值到色彩空間的映射。如果沒有提供，默認設置
annot :如果為True，則將數據值寫入每個單元格中

grid.map_upper() # 上半部
grid.map_diag() # 對角線
grid.map_lower() # 下半部
```
實用連結：<br>
[numpy.random.uniform均匀分布](https://blog.csdn.net/weixin_41770169/article/details/80740370)<br>
[Heatmap](https://www.jianshu.com/p/363bbf6ec335)<br>
[Pairplot應用實例](https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166)

---

### **Day_016_HW** (2019-05-01) － 模型初體驗 Logistic Regression：

---

### **Day_017_HW** (2019-05-02) － 特徵工程簡介：

---

### **Day_018_HW** (2019-05-03) － ：

---

### **Day_019_HW** (2019-05-04) － 數值型特徵-補缺失值與標準化：

---

### **Day_020_HW** (2019-05-05) － ：

---

