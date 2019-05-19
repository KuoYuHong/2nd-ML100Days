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
Label encoding: 把每個類別 mapping 到某個整數，不會**增加新欄位**
One Hot encoding: 為每個類別新增一個欄位，用 0/1 表示是否

實用連接：<br>
[Label Encoder vs. One Hot Encoder in Machine Learning](https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
)<br>
[機器學習中的 Label Encoder和One Hot Encoder](https://kknews.cc/zh-tw/other/kba3lvv.html)

第一次接觸花了一些時間理解

---

### **Day_005_HW** (2019-04-20) － EDA資料分佈：

難易度：:star::star:

針對自己有興趣的欄位觀察資料分布、**畫出直方圖**，在畫圖花了較多時間操作和理解，**有許多客製化方法**決定圖的樣子
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

觀察資料當中**可能有outlier的欄位**、解釋可能的原因，把篩選欄位印出圖表，
在理解ECDF(Emprical Cumulative Density Plot)的地方較久，對於**檢查/處理異常值**的操作要再熟悉些

實用連結：<br>
[Ways to Detect and Remove the Outliers](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba)<br>
[ECDF with Python](https://stackoverflow.com/questions/14006520/ecdf-in-python-without-step-function)

---

### **Day_007_HW** (2019-04-22) － 常用的數值取代：中位數與分位數連續數值標準化：

難易度：:star::star::star:

對**NA值**用不同方法**進行填補**，以及數值標準化

實用連結：<br>
[Is it a good practice to always scale/normalize data for machine learning?](https://stats.stackexchange.com/questions/189652/is-it-a-good-practice-to-always-scale-normalize-data-for-machine-learning)

---

### **Day_008_HW** (2019-04-23) － DataFrame operationData frame merge/常用的 DataFrame 操作：

難易度：:star::star::star:

使用**pd.cut方法將資料分組**，並**畫出箱型圖**觀察
```
pd.cut()等寬劃分 #每一組組距一樣<br>
pd.qcut()等頻劃分 #每一組會出現的頻率一樣
```
實用連結：<br>
[pandas的cut&qcut函數](https://medium.com/@morris_tai/pandas%E7%9A%84cut-qcut%E5%87%BD%E6%95%B8-93c244e34cfc)

---

### **Day_009_HW** (2019-04-24) － 程式實作 EDA: correlation/相關係數簡介：

難易度：:star:

熟悉**相關係數**，以randint和normal方式隨機產生數值**畫出scatter plot圖表**

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

當使用的圖表**看不出規律或如何解讀**時，使用**不同的方式呈現圖表**<br>
找出自己想要的欄位**進行資料處理時花費較多時間**，摸索了一下
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

針對年齡分組與排序，**畫出KDE圖**和**長條圖**<br>
在**使用seaborn套件畫KDE圖**時花較多時間摸索

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

熟悉**數值的離散化**的調整工具<br>
複習**pd.cut()函數**，這在作業中出現過滿多次了

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

**離散化自己有興趣的欄位**，有些欄位比較難弄，弄了滿多時間，參數一直調不好:joy:<br>
需要參考到前面Day11、12的方法，也算是在複習前面的部分<br>
有許多觀念會打結在一起，需要能夠融會貫通才行

實用連結：<br>
[seaborn入門（二）：barplot與countplot](https://zhuanlan.zhihu.com/p/24553277)

---

### **Day_014_HW** (2019-04-29) － Subplots：

難易度：:star::star:

使用**subplot**在不同位置畫圖形出來，同樣也是要**觀察資料、清理NA值**<br>
有許多好用的畫圖方法，需要來好好研究

實用連結：<br>
[Subplot](https://matplotlib.org/examples/pylab_examples/subplots_demo.html)<br>
[Seaborn.jointplot](https://seaborn.pydata.org/generated/seaborn.jointplot.html)

---

### **Day_015_HW** (2019-04-30) － Heatmap & Grid-plot：

難易度：:star::star:

練習使用**random函數**，有滿多種沒用過的方法<br>
之前曾經練習過subplot也複習了一下，一些**新的畫圖方法**也要熟記
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

難易度：:star:

複習前面運用到的方法，產生資料結果<br>
主要讓我們**熟悉參加Kaggle競賽**並**提交作業**，為之後期中、期末考做準備

```
Scikit-learn 預處理工具 Imputer：
可將缺失值替換為均值、中位數、眾數三種
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
```

實用連結：<br>
[Python 機器學習 Scikit-learn 完全入門指南](https://kknews.cc/zh-tw/other/g5qoogm.html)

---

> ### 完成了第一部分資料清理數據前處理，不過這只是學習ML的剛開始，要做好基本功、持續努力

## 主題二：資料科學特徵工程技術

### **Day_017_HW** (2019-05-02) － 特徵工程簡介：

難易度：:star::star:

辨識**特徵工程樣貌**以及**類別型欄位**、**目標值**，對於**特徵工程的觀察與處理手法**有初步的認識

```
使用display取代print：(限用於Ipython)
>>>item = "AAAAA"
>>>display(item)
'AAAAA'

也可用Image直接印出圖片
```

實用連結：<br>
[知乎 - 特徵工程到底是什麼](https://www.zhihu.com/question/29316149)<br>
[痞客幫-iT邦2019鐵人賽 : 為什麼特徵工程很重要](https://ithelp.ithome.com.tw/articles/10200041?sc=iThelpR)<br>
[log1p(x) 和expm1(x) 函數的實現](https://blog.csdn.net/liyuanbhu/article/details/8544644)<br>
[Ipython.display顯示圖像問題](https://bbs.csdn.net/topics/392144095)

---

### **Day_018_HW** (2019-05-03) － 特徵類型：

難易度：:star:

對各種資料類型進行操作來觀察，**數值與類別型特徵**的處理手法

```
groupby("item") #將資料依照自己要的column分組

aggregate(min,np.mean,max) #聚合多個欄位在一起

shape[0] #讀取矩陣第一維度的長度

在print當中使用：f ' { 變數 } 會印出的文字 '
可讓裡面的字串不用打"或'，字串外面可不用+來連結，可用{}裡面塞變數隔開來
```

實用連結：<br>
[Python Tutorial 第二堂 - 數值與字串型態](https://openhome.cc/Gossip/CodeData/PythonTutorial/NumericStringPy3.html)<br>
[Python3.7.2 : Built-in Types Python 官方說明](https://docs.python.org/3/library/stdtypes.html)<br>
[python: numpy--函數shape用法](https://blog.csdn.net/xingchengmeng/article/details/62881859)

---

### **Day_019_HW** (2019-05-04) － 數值型特徵-補缺失值與標準化：

難易度：:star::star:

進行一些**缺失值的填補**，思考**填補方法**，熟悉各種填補情況下搭配使用**線性回歸**(Linear regression)和**羅吉斯迴歸分析**(Logistic regression)的效果表現<br>
**標準化/最小最大化**使用上的差異

實用連結：<br>
[掘金 : Python數據分析基礎 : 數據缺失值處理](https://juejin.im/post/5b5c4e6c6fb9a04f90791e0c)

---

### **Day_020_HW** (2019-05-06)(似乎改成一週只出六天作業:joy:) － 數值型特徵 - 去除離群值：

難易度：:star::star::star:

了解離群職出現會有甚麼問題？<br>
去除離群值：**捨棄**/**調整離群值**做法<br>
刪除前最好能先**了解該數值會離群的可能原因**

```
log1p：
對數去偏就是使用自然對數去除偏態
常見於計數 / 價格這類非負且可能為 0 的欄位
因為需要將 0 對應到 0，所以先加一 (plus one) 再取對數 (log)
還原時使用 expm1，也就是先取指數 (exp) 後再減一 (minus one)

sns.regplot #透過圖表顯示回歸關係，點圖+回歸線

df.clip(a,b) #將df中的數字侷限在a~b之間

sklearn cross_val_score() #驗證用來評分資料準確度
```

實用連結：<br>
[離群值! 離群值? 離群值!](https://zhuanlan.zhihu.com/p/33468998)<br>
[log1p(x) 和expm1(x) 函數的實現](https://blog.csdn.net/liyuanbhu/article/details/8544644)<br>
[機器學習：交叉驗證](https://ithelp.ithome.com.tw/articles/10197461)

---

### **Day_021_HW** (2019-05-07) － 數值型特徵 - 去除偏態：

難易度：:star::star:

了解去除偏態手法：(Skewness)<br>
1.標準化平移、去離群值<br>
2.開根號乘以10<br>
3.對數去偏後的新分布，平均值就比較具有代表性

當**離群資料比例例太高**，或者**平均值沒有代表性**時，可以考慮**去除偏態**<br>
去除偏態包含 : **對數去偏**、**方根去偏**以及**分布去偏**<br>
使用 box-cox 分布去偏時，除了注意**λ參數要介於 0 到 0.5 之間**，並且要注意**轉換前的數值不可小於等於 0**

```
deepcopy #深拷貝：新對象的值不會因為原對象的改變而改變
```

實用連結：<br>
[機器學習數學|偏度與峰度及其python 實現](https://blog.csdn.net/u013555719/article/details/78530879)

---

### **Day_022_HW** (2019-05-08) － 類別型特徵 - 基礎處理：

難易度：:star::star::star:

了解調整標籤編碼(Label Encoder) / 獨熱編碼 (One Hot Encoder) 方式，對於**線性迴歸**以及**梯度提升樹**兩種模型，何者**影響比較大**?對**預測結果**有何影響?<br>

類別型特徵**建議預設採用標籤編碼**，除非該特徵重要性高，且可能值較少(獨熱編碼時負擔較低) 時，才應考慮使用獨熱編碼<br>
**獨熱編碼缺點**是需要**較大的記憶空間與計算時間**，且**類別數量越多時越嚴重**<br>

類別型特徵有**標籤編碼 (Label Encoding)**與**獨熱編碼(One Hot Encoding)**兩種基礎編碼方式<br>
兩種編碼中**標籤編碼比較常用**，當特徵重要性高，且可能值較少時，才應該考慮獨熱編碼

```
GradientBoostingRegressor #梯度提升樹

df_temp = pd.DataFrame() #設立一個空值DF

標籤編碼：LabelEncoder
獨熱編碼：get_dummies(sparse=False,dummy_na=False,drop_first=False)
sparse(稀疏)：
虛擬列是否應該稀疏。如果數據是Series或者包含所有列，則返回ReflectionDataFrame。否則返回帶有一些SparseBlock的DataFrame
dummy_na : 
增加一列表示空缺值，如果False就忽略空缺值
drop_first : 
獲得k中的k-1個類別值，去除第一個
```

實用連結：<br>
[數據預處理：獨熱編碼（One-Hot Encoding）和 LabelEncoder標籤編碼](https://www.twblogs.net/a/5baab6e32b7177781a0e6859/zh-cn/)<br>
[sklearn中的gbt(gbdt/gbrt)](http://d0evi1.com/sklearn/gbdt/)

---

### **Day_023_HW** (2019-05-09) － 類別型特徵 - 均值編碼：

難易度：:star::star:

當類別特徵與目標明顯相關時，該考慮採用均值編碼<br>
均值編碼最大的問題在於相當容易 Overfitting<br>
平滑化的方式能修正均值編碼容易 Overfitting 的問題，但效果有限，因此仍須經過檢驗後再決定是否該使用均值編碼

```
pd.merge(data, mean_df, how='left') # 使用 how='left' 是完全保留原資料表的所有 index 與順序

data.drop([c] , axis=1) #刪除觀測值或欄位，axis = 0 刪除row，axis = 1 刪除column
```

實用連結：<br>
[平均數編碼 ：針對高基數定性特徵(類別特徵)的數據處理/ 特徵工程](https://zhuanlan.zhihu.com/p/26308272)

---

### **Day_024_HW** (2019-05-10) － 類別型特徵 - 其他進階處理：

難易度：:star::star:

計數編碼：<br>
計數編碼是計算類別在資料中的出現次數，當目標平均值與類別筆數呈正/負相關時，可以考慮使用

雜湊編碼：<br>
相異類別的數量量非常龐大時，特徵雜湊是一種折衷方案<br>
在計算空間/時間與鑑別度間取折衷，也提高了了訊息密度，減少無用的標籤

```
計數編碼：
count_df = df.groupby(['Ticket'])['Name'].agg({'Ticket_Count':'size'}).reset_index()

雜湊編碼：
df_temp['Ticket_Hash'] = df['Ticket'].map(lambda x:hash(x) % 10)
```

實用連結：<br>
[Feature hashing (特徵哈希)](https://blog.csdn.net/laolu1573/article/details/79410187)<br>
[基於sklearn的文本特徵抽取](https://www.jianshu.com/p/063840752151)

---

### **Day_025_HW** (2019-05-11) － 時間型特徵：

難易度：:star::star:

時間的週期概念：<br>
年週期與春夏秋冬季節溫度相關<br>
月週期與薪水、繳費相關<br>
周週期與周休、消費習慣相關<br>
日週期與生理理時鐘相關<br>

時間型特徵最常用的是特徵分解 - 拆解成年/月/日/時/分/秒的分類值

實用連結：<br>
[PYTHON-基礎-時間日期處理小結](http://www.wklken.me/posts/2015/03/03/python-base-datetime.html)<br>
[datetime — Basic date and time types](https://docs.python.org/3/library/datetime.html)

---

### **Day_026_HW** (2019-05-13) － 特徵組合 - 數值與數值組合：

難易度：:star::star::star:

在了解**經緯度一圈長度比**的地方花了比較久，經緯度數值確實要比較接近真實情況加入評估較好，但實際特徵評估結果卻略差一些些，有可能對於橢圓的地球來說在不同經緯度位置效果會不同

```
經緯度一圈的長度比：cos(40.75度) : 1 = 0.75756 : 1
latitude_average = df['pickup_latitude'].mean()
latitude_factor = math.cos(latitude_average/180math.pi)
df['distance_real'] = ((df['longitude_diff']latitude_factor)**2 + df['latitude_diff']**2)**0.5
```

實用連結：<br>
[特徵組合&特徵交叉 (Feature Crosses)](https://segmentfault.com/a/1190000014799038)<br>
[簡單高效的組合特徵自動挖掘框架](https://zhuanlan.zhihu.com/p/42946318)<br>
[經緯度與公里的計算](http://wp.mlab.tw/?p=2200)

---

### **Day_027_HW** (2019-05-14) － 特徵組合 - 類別與數值組合：

難易度：:star::star::star:

群聚編碼：<br>
數值型特徵對文字型特徵最重要的特徵組合方式<br>
常見的有 mean, median, mode, max, min, count 等<br>
![均值編碼&群聚編碼比較](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E5%9D%87%E5%80%BC%E7%B7%A8%E7%A2%BC%26%E7%BE%A4%E8%81%9A%E7%B7%A8%E7%A2%BC%E6%AF%94%E8%BC%83.png)

均值編碼容易overfitting/聚類編碼不容易overfitting的<br>
不過類別型和數值型欄位的選用會影響很大，如何提高**生存率預估**就要選擇與他最相關的欄位來做，效果較好

實用連結：<br>
[利用 Python 數據分析之數據聚合與分組](https://zhuanlan.zhihu.com/p/27590154)

---

### **Day_028_HW** (2019-05-15) － 特徵選擇：

難易度：:star::star::star::star:

特徵選擇有三大類方法：<br>
•過濾法 (Filter) : 選定統計數值與設定門檻，刪除低於門檻的特徵<br>
•包裝法 (Wrapper) : 根據目標函數，逐步加入特徵或刪除特徵<br>
•嵌入法 (Embedded) : 使用機器學習模型，根據擬合後的係數，刪除係數低於門檻的特徵<br>
本日內容將會介紹三種較常用的特徵選擇法：<br>
•過濾法 : 相關係數過濾法<br>
•嵌入法 : L1(Lasso)嵌入法，GDBT(梯度提升樹)嵌入法

```
from itertools import compress：
compress 可用於對數據進行篩選，當selectors 的某個元素為true 時，則保留data 對應位置的元素，否則去除

L1_mask = list((L1_Reg.coef_>0) | (L1_Reg.coef_<0))
L1_list = list(compress(list(df), list(L1_mask))) #將df轉化成list，只保留要的相關係數之間的數值
L1_list
```

調整過許多次參數<br>
今天有比較多觀念在裡面，要綜合前幾天所學的去做**判斷、調整**，需要多去理解

實用連結：<br>
[特徵選擇](https://zhuanlan.zhihu.com/p/32749489)<br>
[特徵選擇線上手冊](https://machine-learning-python.kspax.io/intro-1)

---

### **Day_029_HW** (2019-05-16) － 特徵評估：

難易度：:star::star::star:

特徵重要性評估之後進行組合，看是否能夠提高生存率預估正確率，需要具備領域知識才有可能進一步提高正確率<br>

實用連結：<br>
[機器學習 - 特徵選擇算法流程、分類、優化與發展綜述](https://juejin.im/post/5a1f7903f265da431c70144c)<br>
[Permutation Importance](https://www.kaggle.com/dansbecker/permutation-importance?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights)

---

### **Day_030_HW** (2019-05-17) － 分類型特徵優化 - 葉編碼：

難易度：:star::star::star:

了解葉編碼的使用，有時候效果不是很明顯，還是要觀看資料的本質後做出適當的特徵工程方法<br>

實用連結：<br>
[Feature transformations with ensembles of trees](https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#example-ensemble-plot-feature-transformation-py)<br>
[CTR 預估[十一]： Algorithm-GBDT Encoder](https://zhuanlan.zhihu.com/p/31734283)<br>
[三分鐘了解推薦系統中的分解機方法(Factorization Machine, FM)](https://kknews.cc/zh-tw/other/62k4rml.html)

---
> ### 

## 主題三：機器學習基礎模型建立

### **Day_031_HW** (2019-05-18) － 機器學習概論：

難易度：:star:

了解機器學習、透過一些影片和文獻讓我們更了解<br>

實用連結：<br>
[了解機器學習/人工智慧的各種名詞意義](https://kopu.chat/2017/07/28/%E6%A9%9F%E5%99%A8%E6%98%AF%E6%80%8E%E9%BA%BC%E5%BE%9E%E8%B3%87%E6%96%99%E4%B8%AD%E3%80%8C%E5%AD%B8%E3%80%8D%E5%88%B0%E6%9D%B1%E8%A5%BF%E7%9A%84%E5%91%A2/)<br>
[聽聽人工智慧頂尖學者分享 AI 的知識](https://www.ted.com/talks/fei_fei_li_how_we_re_teaching_computers_to_understand_pictures?language=zh-tw)

---

### **Day_032_HW** (2019-05-19) － 機器學習-流程與步驟：

難易度：:star:

今天也是讓我們閱讀一些機器學習相關文章來思考文章要表達的目標和方法<br>

實用連結：<br>
ML 流程 by Google-[The 7 Steps of Machine Learning (AI Adventures)](https://www.youtube.com/watch?v=nKW8Ndu7Mjw)

---

### **Day_033_HW** (2019-05-20) － ：

---

### **Day_034_HW** (2019-05-21) － ：

---

### **Day_035_HW** (2019-05-22) － ：

---

### **Day_036_HW** (2019-05-23) － ：

---

### **Day_037_HW** (2019-05-24) － ：

---





