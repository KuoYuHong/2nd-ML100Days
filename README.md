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

### **Day_001_HW** － 資料介紹與評估資料：
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

### **Day_002_HW** － EDA-1/讀取資料EDA: Data summary：
難易度：:star:

這次作業主要用到[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)的資料集

作業主要練習資料的row和col、欄位、顯示資料等等的熟悉

---

### **Day_003_HW** － 3-1如何新建一個 dataframe?3-2 如何讀取其他資料? (非 csv 的資料)：
難易度：:star::star:

練習**DataFrame的操作**以及檔案的存儲，日後會有非常多機會操作Pandas的DF

[Pickle檔案](https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/373209/
)
：實現python物件的**序列化和反序列化**

讀取圖片的話以前曾經用過**PIL套件**，使用提供的資料經過資料處理後弄一些圖片出來，難易算是普通

---

### **Day_004_HW** － EDA: 欄位的資料類型介紹及處理：

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

### **Day_005_HW** － EDA資料分佈：

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

### **Day_006_HW** － EDA: Outlier 及處理：

難易度：:star::star::star::star:

觀察資料當中**可能有outlier的欄位**、解釋可能的原因，把篩選欄位印出圖表，
在理解ECDF(Emprical Cumulative Density Plot)的地方較久，對於**檢查/處理異常值**的操作要再熟悉些

實用連結：<br>
[Ways to Detect and Remove the Outliers](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba)<br>
[ECDF with Python](https://stackoverflow.com/questions/14006520/ecdf-in-python-without-step-function)

---

### **Day_007_HW** － 常用的數值取代：中位數與分位數連續數值標準化：

難易度：:star::star::star:

對**NA值**用不同方法**進行填補**，以及數值標準化

實用連結：<br>
[Is it a good practice to always scale/normalize data for machine learning?](https://stats.stackexchange.com/questions/189652/is-it-a-good-practice-to-always-scale-normalize-data-for-machine-learning)

---

### **Day_008_HW** － DataFrame operationData frame merge/常用的 DataFrame 操作：

難易度：:star::star::star:

使用**pd.cut方法將資料分組**，並**畫出箱型圖**觀察
```
pd.cut()等寬劃分 #每一組組距一樣<br>
pd.qcut()等頻劃分 #每一組會出現的頻率一樣
```
實用連結：<br>
[pandas的cut&qcut函數](https://medium.com/@morris_tai/pandas%E7%9A%84cut-qcut%E5%87%BD%E6%95%B8-93c244e34cfc)

---

### **Day_009_HW** － 程式實作 EDA: correlation/相關係數簡介：

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

### **Day_010_HW** － EDA from Correlation：

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

### **Day_011_HW** － EDA: 不同數值範圍間的特徵如何檢視/繪圖與樣式Kernel Density Estimation (KDE)：

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

### **Day_012_HW** － EDA: 把連續型變數離散化：

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

### **Day_013_HW** － 程式實作 把連續型變數離散化：

難易度：:star::star::star::star:

**離散化自己有興趣的欄位**，有些欄位比較難弄，弄了滿多時間，參數一直調不好:joy:<br>
需要參考到前面Day11、12的方法，也算是在複習前面的部分<br>
有許多觀念會打結在一起，需要能夠融會貫通才行

實用連結：<br>
[seaborn入門（二）：barplot與countplot](https://zhuanlan.zhihu.com/p/24553277)

---

### **Day_014_HW** － Subplots：

難易度：:star::star:

使用**subplot**在不同位置畫圖形出來，同樣也是要**觀察資料、清理NA值**<br>
有許多好用的畫圖方法，需要來好好研究

實用連結：<br>
[Subplot](https://matplotlib.org/examples/pylab_examples/subplots_demo.html)<br>
[Seaborn.jointplot](https://seaborn.pydata.org/generated/seaborn.jointplot.html)

---

### **Day_015_HW** － Heatmap & Grid-plot：

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

### **Day_016_HW** － 模型初體驗 Logistic Regression：

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

### **Day_017_HW** － 特徵工程簡介：

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

### **Day_018_HW** － 特徵類型：

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

### **Day_019_HW** － 數值型特徵-補缺失值與標準化：

難易度：:star::star:

進行一些**缺失值的填補**，思考**填補方法**，熟悉各種填補情況下搭配使用**線性回歸**(Linear regression)和**羅吉斯迴歸分析**(Logistic regression)的效果表現<br>
**標準化/最小最大化**使用上的差異

實用連結：<br>
[掘金 : Python數據分析基礎 : 數據缺失值處理](https://juejin.im/post/5b5c4e6c6fb9a04f90791e0c)

---

### **Day_020_HW**(似乎改成一週只出六天作業:joy:) － 數值型特徵 - 去除離群值：

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

### **Day_021_HW** － 數值型特徵 - 去除偏態：

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

### **Day_022_HW** － 類別型特徵 - 基礎處理：

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

### **Day_023_HW** － 類別型特徵 - 均值編碼：

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

### **Day_024_HW** － 類別型特徵 - 其他進階處理：

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

### **Day_025_HW** － 時間型特徵：

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

### **Day_026_HW** － 特徵組合 - 數值與數值組合：

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

### **Day_027_HW** － 特徵組合 - 類別與數值組合：

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

### **Day_028_HW** － 特徵選擇：

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

### **Day_029_HW** － 特徵評估：

難易度：:star::star::star:

特徵重要性評估之後進行組合，看是否能夠提高生存率預估正確率，需要具備領域知識才有可能進一步提高正確率<br>

實用連結：<br>
[機器學習 - 特徵選擇算法流程、分類、優化與發展綜述](https://juejin.im/post/5a1f7903f265da431c70144c)<br>
[Permutation Importance](https://www.kaggle.com/dansbecker/permutation-importance?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights)

---

### **Day_030_HW** － 分類型特徵優化 - 葉編碼：

難易度：:star::star::star:

了解葉編碼的使用，有時候效果不是很明顯，還是要觀看資料的本質後做出適當的特徵工程方法<br>

實用連結：<br>
[Feature transformations with ensembles of trees](https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#example-ensemble-plot-feature-transformation-py)<br>
[CTR 預估[十一]： Algorithm-GBDT Encoder](https://zhuanlan.zhihu.com/p/31734283)<br>
[三分鐘了解推薦系統中的分解機方法(Factorization Machine, FM)](https://kknews.cc/zh-tw/other/62k4rml.html)

---
> ### 完成了第二部分特徵工程學習，掌握特徵的取捨才能讓我們得到良好的特徵結果

## 主題三：機器學習基礎模型建立

### **Day_031_HW** － 機器學習概論：

難易度：:star:

了解機器學習、透過一些影片和文獻讓我們更了解<br>

實用連結：<br>
[了解機器學習/人工智慧的各種名詞意義](https://kopu.chat/2017/07/28/%E6%A9%9F%E5%99%A8%E6%98%AF%E6%80%8E%E9%BA%BC%E5%BE%9E%E8%B3%87%E6%96%99%E4%B8%AD%E3%80%8C%E5%AD%B8%E3%80%8D%E5%88%B0%E6%9D%B1%E8%A5%BF%E7%9A%84%E5%91%A2/)<br>
[聽聽人工智慧頂尖學者分享 AI 的知識](https://www.ted.com/talks/fei_fei_li_how_we_re_teaching_computers_to_understand_pictures?language=zh-tw)

---

### **Day_032_HW** － 機器學習-流程與步驟：

難易度：:star:

今天也是讓我們閱讀一些機器學習相關文章來思考文章要表達的目標和方法<br>

實用連結：<br>
ML 流程 by Google-[The 7 Steps of Machine Learning (AI Adventures)](https://www.youtube.com/watch?v=nKW8Ndu7Mjw)

---

### **Day_033_HW** － 機器如何學習?：

難易度：:star:

思考了何謂機器學習與過擬合、模型的泛化能力、分類問題與回歸問題分別可用的目標函數有哪些後，讓我們了解到機器學習概論比較深的理論，學好前面的觀念，後來在實際操作時才會比較順利

實用連結：<br>
理解機器學習中很重要的 Bias/Variance trade-off 的意義為何？<br>
[機器學習老中醫：利用學習曲線診斷模型的偏差和方差](http://bangqu.com/yjB839.html)<br>
[機器學習中的目標函數分析](https://www.twblogs.net/a/5c188f10bd9eee5e41847a50)

---

### **Day_034_HW** － 訓練/測試集切分的概念：

難易度：:star::star::star::star:

了解sklearn使用train_test_split、KFold的訓練/測試集切分方法，並使用train_test_split切分資料，有些參數設定要理解一下，摸索比較久<br>

[sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)<br>
[sklearn.model_selection.KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold)<br>

```
X_train,X_test, y_train, y_test =cross_validation.train_test_split(train_data,train_target,test_size=0.4, random_state=0)
引數解釋：
train_data：所要劃分的樣本特徵集
train_target：所要劃分的樣本結果
test_size：樣本佔比，如果是整數的話就是樣本的數量
random_state：是隨機數的種子。是該組隨機數的編號，在需要重複試驗的時候，保證得到一組一樣的隨機數。比如你每次都填1，其他引數一樣的情況下你得到的隨機陣列是一樣的。但填0或不填，每次都會不一樣。
隨機數的產生取決於種子，隨機數和種子之間的關係遵從以下兩個規則：種子不同，產生不同的隨機數；種子相同，即使例項不同也產生相同的隨機數。
```

實用連結：<br>
理解訓練、驗證與測試集的意義與用途<br>
[台大電機李宏毅教授講解訊練/驗證/測試集的意義](https://www.youtube.com/watch?v=D_S6y0Jm6dQ&feature=youtu.be&t=1948)

---

### **Day_035_HW** － regression vs. classification：

難易度：:star:

了解多分類問題與多標籤問題差別，試著去辨認出來<br>

實用連結：<br>
了解回歸與分類的差異在哪裡?<br>
[回歸與分類的比較](http://zylix666.blogspot.com/2016/06/supervised-classificationregression.html)

---

### **Day_036_HW** － 評估指標選定/evaluation metrics：

難易度：:star::star::star:

了解評估指標的選定，實際上操作分類問題的評估指標有點卡，要再熟悉<br>
```
Accuracy = 正確分類樣本數/總樣本數

評估指標 - 迴歸：
觀察「預測值」 (Prediction) 與「實際值」 (Ground truth) 的差距
MAE、MSE、R-square

評估指標 - 分類：
觀察「預測值」 (prediction) 與「實際值」 (Ground truth) 的正確程度
AUC(Area Under Curve)、F1 - Score (Precision, Recall)
```
實用連結：<br>
深入了解超常用的指標 AUC：<br>
[超詳細解說 AUC (英文)](https://www.dataschool.io/roc-curves-and-auc-explained/)<br>

學習更多評估指標，來衡量機器學習模型的準確度：<br>
[更多評估指標](https://zhuanlan.zhihu.com/p/30721429)<br>

---

### **Day_037_HW** － regression model 介紹 - 線性迴歸/羅吉斯回歸：

難易度：:star:

對於線性迴歸/羅吉斯回歸做深入了解，前面的作業也有操作過這些迴歸，現在再讓觀念更清晰<br>

實用連結：<br>
[超人氣 Stanford 教授 Andrew Ng 教你 Linear regression](https://zh-tw.coursera.org/lecture/machine-learning/model-representation-db3jS)(強烈推薦觀看) <br>
[Logistic regression 數學原理](https://blog.csdn.net/qq_23269761/article/details/81778585)<br>

---

### **Day_038_HW** － regression model 程式碼撰寫：

難易度：:star::star:

使用sklearn中datasets資料集練習Linear Regression和Logistic Regression<br>
```
Scikit-learn 中的 Logistic Regression 參數：
Penalty : “L1” , “L2”。使用 L1 或 L2 的正則化參數
C : 正則化的強度，數字越小，模型越簡單
Solver : 對損失函數不同的優化方法
Multi-class : 選擇 one-vs-rest 或 multi-nominal 分類方式，當目標是multi-class  時要特別注意，若有 10 個 class， ovr 是訓練 10 個二分類模型，第一個模型負責分類 (class1, non-class1) ；第二個負責(class2, non-class2) ，以此類推。multi-nominal  是直接訓練多分類模型
```
實用連結：<br>
[超多 Linear Regression / Logistic Regression 的 examples](https://github.com/trekhleb/homemade-machine-learning)<br>
[深入了解 multinomial Logistic Regression 的原理](http://dataaspirant.com/2017/05/15/implement-multinomial-logistic-regression-python/)<br>

---

### **Day_039_HW** － regression model 介紹 - LASSO 回歸/ Ridge 回歸：

難易度：:star::star:

了解Ridge Regression以及Lasso Regression的區別以及做法<br>
```
機器學習模型的目標函數中有兩個非常重要的元素
•損失函數 (Loss function)
•正則化 (Regularization)
損失函數衡量預測值與實際值的差異，讓模型能往正確的方向學習
正則化則是避免模型變得過於複雜，造成過擬合 (Over-fitting)
```
---

### **Day_040_HW** － regression model 程式碼撰寫：

難易度：:star::star:

了解LASSO與Ridge概念後實際操作資料集，觀看不同參數下的狀況與線性迴歸做對比<br>

LASSO與Ridge的結果並沒有比原本的線性回歸來得好，這是因為目標函數被加上了正規化函數，讓模型不能過於複雜，相當於限制模型擬和資料的能力。因此若沒有發現 Over-fitting 的情況，是可以不需要一開始就加上太強的正規化<br>

---

### **Day_041_HW** － tree based model - 決策樹 (Decision Tree) 模型介紹：

難易度：:star:

了解決策樹原理及相關知識<br>

決策樹在資料分布明顯的狀況下,有機會在訓練時將 training loss 完全降成 0<br>

決策樹 (Decision Tree)：<br>
從訓練資料中找出規則，讓每一次決策能使訊息增益 (Information Gain) 最大化
訊息增益越大代表切分後的兩群資料，群內相似程度越高<br>

---

### **Day_042_HW** － tree based model - 決策樹程式碼撰寫：

難易度：:star::star:

了解決策樹DecisionTreeClassifier、DecisionTreeRegressor模型的應用

```
決策樹的超參數：
Criterion: 衡量資料相似程度的 
metricMax_depth: 樹能生長的最深限制
Min_samples_split: 至少要多少樣本以上才進行切分
Min_samples_lear: 最終的葉子 (節點) 上至少要有多少樣本
```
實用連結：
[可安裝額外的套件 graphviz，畫出決策樹的圖形幫助理解模型分類的準則](https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176)

---

### **Day_043_HW** － tree based model - 隨機森林 (Random Forest) 介紹：

難易度：:star::star:

了解隨機森林的基本概念，以前曾經接觸過，算是重新複習過一次<br>
```
在 training data 中, 從中取出一些 feature & 部份 data 產生出 Tree (通常是CART)
並且重複這步驟多次, 會產生出多棵 Tree 來
最後利用 Ensemble (Majority Vote) 的方法, 結合所有 Tree, 就完成了 Random Forest

bagging採樣方式：
設定最少要 bagging 出 (k / 2) + 1 的 feature, 才比較有顯著結果, K 為原本的 feature 數量
或者另外一個常見設定是 square(k)
```
---

### **Day_044_HW** － tree based model - 隨機森林程式碼撰寫：

難易度：:star::star:

實際進行隨機森林的操作，觀看用同一資料集下，在不同數量的樹、不同深度所得出的結果有何不同？
```
建立模型 (使用 20 顆樹，每棵樹的最大深度為 4)：
clf = RandomForestClassifier(n_estimators=20, max_depth=4)

隨機森林的模型超參數：
同樣是樹的模型，所以像是 max_depth, min_samples_split 都與決策樹相同
可決定要生成數的數量，越多越不容易過擬和，但是運算時間會變長

fromsklearn.ensemble import RandomForestClassifier #集成模型

clf = RandomForestClassifier(
n_estimators=10, #決策樹的數量量
criterion="gini",
max_features="auto", #如何選取 features
max_depth=10,
min_samples_split=2,
min_samples_leaf=1
)
```
實用連結：
[知名 ML youtuber 教你手刻隨機森林 by Python](https://www.youtube.com/watch?v=QHOazyP-YlM)

---

### **Day_045_HW** － tree based model - 梯度提升機 (Gradient Boosting Machine) 介紹：

難易度：:star:

閱讀XGBoost/Light-GBM、Gradient-boosting相關文章並了解

實用連結：<br>
* [梯度提升機原理 - 中文](https://ifun01.com/84A3FW7.html)<br>
文章中的殘差就是前面提到的 Loss，從範例中了解殘差是如何被修正的<br>
* [XGboost 作者講解原理 - 英文](https://www.youtube.com/watch?v=ufHo8vbk6g4)<br>
了解 XGBoost 的目標函數是什麼，模型是怎麼樣進行優化<br>
* [XGBoost 數學原理 slides - 英文](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)<br>
了解 XGBoost 的目標函數數學推導<br>
* [Kaggle 大師帶你了解梯度提升機原理 - 英文](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)<br>
* [完整的 Ensemble 概念 by 李宏毅教授](https://www.youtube.com/watch?v=tH9FH1DH5n0)<br>
* [深入了解 Gradient-boosting - 英文](https://explained.ai/gradient-boosting/index.html)<br>

---

### **Day_046_HW** － tree based model - 梯度提升機程式碼撰寫：

難易度：:star::star:

了解GradientBoostingClassifier的使用方法並實作

實用連結：<br>
[完整調參數攻略-如何使用 Python 調整梯度提升機的超參數](complete-guide-parameter-tuning-gradient-boosting-gbm)

---

> ### 了解機器學習當中的評估指標、基礎模型與樹狀模型，針對不同的問題類型來選用模型去訓練，才能得到比較好的成果！

## 主題四：機器學習調整參數

### **Day_047_HW** － 超參數調整與優化：

難易度：:star::star::star:

熟悉各種不同超參數調整方法，有助於提升訓練結果<br>

```
超參數調整方法：
窮舉法 (Grid Search)：直接指定超參數的組合範圍，每一組參數都訓練完成，再根據驗證集 (validation) 的結果選擇最佳參數
隨機搜尋 (Random Search)：指定超參數的範圍，用均勻分布進行參數抽樣，用抽到的參數進行訓練，再根據驗證集的結果選擇最佳參數
隨機搜尋通常都能獲得更佳的結果
```

實用連結：<br>
[劍橋實驗室教你如何調參數 -  英文](https://cambridgecoding.wordpress.com/2016/04/03/scanning-hyperspace-how-to-tune-machine-learning-models/)<br>
[教你使用 Python 調整隨機森林參數 - 英文](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)<br>

---

### **Day_048_HW** － Kaggle 競賽平台介紹：

難易度：:star::star::star::star:

在Kaggle的scikit-learn-practice比賽練習，觀看別人的Kernels，試著去理解寫法並內化成自己的知識<br>

實用連結：<br>
[scikit-learn-practice](https://www.kaggle.com/c/data-science-london-scikit-learn)

---

### **Day_049_HW** － 集成方法 : 混合泛化(Blending)：

難易度：:star::star::star::star:

練習集成方法中的混合泛化(Blending)，去觀看結果分數有沒有比原先來得好<br>

實用連結：<br>
機器學習技法 Lecture 7: Blending and Bagging<br>
林軒田老師公開課程[網頁連結](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/207_handout.pdf) [影片連結](https://www.youtube.com/watch?v=mjUKsp0MvMI&list=PLXVfgk9fNX2IQOYPmqjqWsNUFl2kpk1U2&index=27&t=0s)<br>
當我們在網路上自己搜尋 Blending 時，往往搜尋到的都是林軒田老師的課程筆記，因此我們推薦同學如果對於 Blending 或  Bagging 的理論想要一探更完整內容的話，不妨來這邊尋找研讀的資料，相信絕對不會讓您失望 (如果太困難，也可以參考網路上眾多的閱讀筆記)

Superblend<br>
Kaggle 競賽網站-Kernel 範例[網頁連結](https://www.kaggle.com/tunguz/superblend)<br>
這邊就是我們所謂競賽中的 Blending Kernel，只是決定一個權重，將兩個其他的 Kernel 合併成答案檔，就是這場競賽中的最高分 Kernel，我們並不是要鼓勵大家也去這樣去賺分數，而是在告訴大家 : Blending 的簡單，以及 Blending 的具有威力。

---

### **Day_050_HW** － 集成方法 : 堆疊泛化(Stacking)：

難易度：:star::star::star:

練習集成方法中的堆疊泛化(Stacking)，在混合泛化後進一步調整模型，試著讓結果更好<br>

實用連結：<br>
StackingCVClassifier<br>
mlxtrend 官方網站 [網頁連結](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/)<br>
如何在 Kaggle 首戰中進入前 10%
Wille 個人心得 [網頁連結](https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/)<br>

---

### **期中考Day_050~053_HW**：

難易度：:star::star::star::star:

進行百日馬拉松期中考：<br>
#### Overview
隨著移動設備的完善和普及，零售與電子商務進入了高速發展階段，這其中以 O2O（Online to Offline）消費最為熱絡。據統計，O2O 行業估值上億的創業公司至少有 10 家，也不乏百億巨頭的身影。 O2O 行業每日自然流量有數億消費者，各類 APP 每天記錄了超過百億條用戶行為和位置記錄，因而成為大數據科研和商業化運營的最佳結合點之一。

以優惠券活化老用戶或吸引新客戶消費是 O2O 的一種重要營銷方式。然而，隨機投放的優惠券對多數用戶造成無意義的干擾。對商家而言，濫發的優惠券可能降低品牌聲譽，同時難以估算營銷成本。個性化投放是提高優惠券核銷率的重要技術，它可以讓具有一定偏好的消費者得到真正的實惠，同時賦予商家更強的營銷能力。本次練習數據擷取自電商之部分數據，希望各位通過分析建模，預測用戶是否會在規定時間內使用相應優惠券。

#### Data
本賽題提供用戶在2016年1月1日至2016年5月31日之間真實線下消費行為，預測用戶在2016年6月領取優惠券後15天以內的使用情況。

#### Evaluation
本賽題目標是預測投放的優惠券是否在規定時間內核銷。針對此任務及一些相關背景知識，以該用戶使用於某日取得之優惠券核銷預測 AUC（ROC 曲線下面積）作為評價標準。即對將 User_id - Date_received - Coupon_id 為一組計算核銷預測的AUC值，若某使用者於同一日取得多張相同優惠券，則任一張核銷皆為有效核銷。

#### 評分標準
提交檔案內容格式須符合比賽格式規定<br>
Public Leaderboard 系統會對每次的提交結果進行評測<br>
比賽結束後會公布 Private Leaderboard 的結果，並以此結果作為最終排名<br>
最低準確度 AUC 需大於0.63<br>

#### 競賽結束後你可以學會
處理理結構化資料<br>
使用 train / valid data 來了解機器學習模型的訓練情形<br>
使用基礎與常於競賽中使用的機器學習模型<br>
調整模型的超參數來來提升準確率<br>
清楚的說明文件讓別人了解你的成果

競賽連結：<br>
[ml100marathon-02-01](https://www.kaggle.com/c/ml100marathon-02-01)

---

### **Day54** － clustering 1 非監督式機器學習簡介：

難易度：:star::star:

非監督學習是否有可能使用評價函數 (Metric) 來鑑別好壞呢?<br>
(Hint : 可以分為 "有目標值" 與 "無目標值" 兩個方向思考)
> 1.其實非監督評價是很困難的, 光是是否要參考目標值本身就是一大難題  
> 2.也因此非監督模型的評價, 有無目標值都有各自的模型可參考  
> 3.簡單來說就是兩者都可以, 但是無目標值的評價方式似乎更合理一些  
> 4.有目標值時, 評估的方式類似分類問題, 但仍須確定生成的分類與原訂的標籤對照關係  
> 5.無目標值的評價法, 比較接近非監督的性質本身, 同一群內資料越靠近/不同群資料越遠 是其主要考量

實用連結：<br>
[Unsupervised learning：PCA](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/PCA.mp4)<br>
[Scikit-learn unsupervised learning](https://scikit-learn.org/stable/unsupervised_learning.html)

---

### **Day55** － clustering 2 聚類算法：

難易度：:star::star:

當問題不清楚或是資料未有標註的情況下，可以嘗試用分群算法幫助瞭解資料結構，而其中一個方法是運用 K-means 聚類算法幫助分群資料<br>
分群算法需要事先定義群數，因此效果評估只能藉由人爲觀察<br>

實用連結：<br>
[Clustering 影片來源：Statistical Learning YT](https://www.youtube.com/watch?v=aIybuNt9ps4)<br>
[Clustering Means Algorithm 影片來源： [ Machine Learning | Andrew Ng ] YT](https://www.youtube.com/watch?v=jAA2g9ItoAc)<br>
[Unsupervised Machine Learning:Flat Clustering](https://pythonprogramming.net/flat-clustering-machine-learning-python-scikit-learn/)

---

### **Day56** － K-mean 觀察 : 使用輪廓分析：

難易度：:star::star:

輪廓分數是一種同群資料點越近 / 不同群資料點越遠時會越大的分數，除了可以評估資料點分群是否得當，也可以用來來評估分群效果<br>
要以輪廓分析觀察 K -mean，除了可以將每個資料點分組觀察以評估資料點分群是否得當，也可用平均值觀察評估不同 K 值的分群效果

---

### **Day57** － clustering 3 階層分群算法：

難易度：:star::star:

階層式分群在無需定義群數的情況下做資料的分群，而後可以用不同的距離定義方式決定資料群組。<br>
分群距離計算方式有 single-link, complete-link, average-link<br>
概念簡單且容易呈現，但不適合用在大資料

實用連結：<br>
Hierarchical Clustering 影片來源：Statistical Learning YT(https://www.youtube.com/watch?v=Tuuc9Y06tAc)<br>
Example：Breast cancer Microarray study 影片來源：Statistical Learning YT(https://www.youtube.com/watch?v=yUJcTpWNY_o)

---

### **Day58** － 階層分群法 觀察 : 使用 2D 樣版資料集：

難易度：:star::star:

了解2D樣版資料集的設計著重於圖形的人機差異，用途在於讓人眼以非量化的方式評估非監督模型的好壞，也因為非監督問題的類型不同，這類資料集也有分群與流形還原等不同對應類型

---

### **Day59** － dimension reduction 1 降維方法-主成份分析：

難易度：:star::star:

降低維度可以幫助我們壓縮及丟棄無用資訊、抽象化及組合新特徵、呈現高維數據。常用的算法爲主成分分析。<br>
在維度太大發生 overfitting 的情況下，可以嘗試用PCA 組成的特徵來做監督式學習，但不建議一開始就做<br>

實用連結：<br>
[Unsupervised learning 影片來源：Statistical Learning](https://www.youtube.com/watch?v=ipyxSYXgzjQ)<br>
[Further Principal Components 影片來源：Statistical Learning](https://www.youtube.com/watch?v=dbuSGWCgdzw)<br>
[Principal Components Regression 影片來源：Statistical Learning](https://www.youtube.com/watch?v=eYxwWGJcOfw)<br>
[Dimentional Reduction 影片來源 Andrew Ng](https://www.youtube.com/watch?v=rng04VJxUt4)

---

### **Day60** － PCA 觀察 : 使用手寫辨識資料集：

難易度：:star::star:

手寫資料集是改寫自手寫辨識集MNIST的，為了使其適合機器學習，除了將背景統一改為黑底白字的灰階圖案，也將大小統一變更為 28*28

---

### **Day61** － dimension reduction 2 降維方法-T-SNE：

難易度：:star::star:

特徵間爲非線性關係時 (e.g. 文字、影像資料)，PCA很容易 underfitting<br>
t-SNE 對於特徵非線性資料有更好的降維呈現能力<br>

實用連結：<br>
[Visualizing Data using t-SNE 影片來源：GoogleTechTalks YT](https://www.youtube.com/watch?v=RJVL80Gg3lA)<br>
[Unsupervised Learning 影片來源：李弘毅 YT](https://www.youtube.com/watch?v=GBUEjkpoxXc)

---

### **Day62** － t-sne 觀察 : 分群與流形還原：

難易度：:star::star:

流形還原就是在高維度到低維度的對應中，盡量保持資料點之間的遠近關係，沒有資料點的地⽅方，就不列列入考量範圍<br>
除了 t-sne 外，較常見的流形還原還有 Isomap 與LLE (Locally Linear Embedding) 等工具

---

### **Day63** － 神經網路介紹：

難易度：:star::star:

在精簡深度學習的方式上：卷積類神經 (CNN) 採用像素遠近，而遞歸類神經 (RNN) 採用著則是時間遠近，那麼，既然有著類似的設計精神，兩者是否有可能互換應用呢?

實用連結：<br>
[人工智慧大歷史 林守德教授演講  / Mora Chen 筆記](https://medium.com/@suipichen/%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E5%A4%A7%E6%AD%B7%E5%8F%B2-ffe46a350543)<br>
[泛科學：3 分鐘搞懂深度學習到底在深什麼 節錄李宏毅老師演講](https://panx.asia/archives/53209)

---

### **Day64** － 深度學習體驗 : 模型調整與學習曲線：

難易度：:star:

了解並操作深度學習體驗平台：TensorFlowPlayGround<br>
TensorFlow PlayGround 是 Google 精心開發的體驗網頁，提供學習者在接觸語言之前，就可以對深度學習能概略了解<br>

實用連結：<br>
[TensorFlowPlayGround平台網址](https://playground.tensorflow.org)<br>
[中文版 TF PlayGround 科技部AI普適研究中心](https://pairlabs.ai/tensorflow-playground/index_tw.html#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.81970&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

---

### **Day65** － 深度學習體驗 : 啟動函數與正規化：

難易度：:star:

操作深度學習體驗平台：TensorFlowPlayGround，理解批次大小 (Batch size) 與學習速率 (Learnig Rate) 對學習結果的影響<br>
經由實驗，體驗不同啟動函數的差異性<br>
體驗正規化 (Regularization) 對學習結果的影響<br>

實用連結：<br>
[Understanding neural networks with TensorFlow Playground Google Cloud 官方教學](https://cloud.google.com/blog/products/gcp/understanding-neural-networks-with-tensorflow-playground)<br>
[深度深度學習網路調參技巧 with TensorFlow Playground 知乎  作者：煉丹實驗室](https://zhuanlan.zhihu.com/p/24720954)

---

### **Day66** － Keras 安裝與介紹：

難易度：:star:

初步了解深度學習套件 : Keras<br>
知道如何在自己的作業系統上安裝 Keras<br>

實用連結：<br>
Keras文檔：<br>
[Github連結](https://github.com/keras-team/keras/tree/master/docs)<br>
Keras: 中文文檔<br>
[連結](https://keras.io/zh/#keras_1)

---

### **Day67** － Keras Dataset：

難易度：:star::star:

了解Keras自帶數據集與模型<br>
以及如何使用 CIFAR10 做類別預測<br>

實用連結：<br>
[Keras: The Python Deep Learning library](https://github.com/keras-team/keras/)<br>
[Keras datase](https://keras.io/datasets/)<br>
[Predicting Boston House Prices](https://www.kaggle.com/sagarnildass/predicting-boston-house-prices)<br>

其餘公開數據集介紹：<br>
Imagenet數據集有1400多萬幅圖片，涵蓋2萬多個類別；其中有超過百萬的圖片有明確的類別標註和圖像中物體位置的標註，具體信息如下：<br>
1）Total number of non-empty synsets : 21841 <br>
2）Total number of images: 14,197,122 <br>
3）Number of images with bounding box annotations: 1,034,908 <br>
4）Number of synsets with SIFT features: 1000 <br>
5）Number of images with SIFT features: 1.2 million<br>
Imagenet數據集是目前深度學習圖像領域應用得非常多的一個領域，關於圖像分類、定位、檢測等研究工作大多基於此數據集展開。Imagenet數據集文檔詳細，有專門的團隊維護，使用非常方便，在計算機視覺領域研究論文中應用非常廣，幾乎成為了目前深度學習圖像領域算法性能檢驗的“標準”數據集。數據集大小：~1TB（ILSVRC2016比賽全部數據）<br>
[下載地址](http://www.image-net.org/about-stats)

COCO：<br>
COCO(Common Objects in Context)是一個新的圖像識別、分割和圖像語義數據集，它有如下特點：<br>
1）Object segmentation <br>
2）Recognition in Context <br>
3）Multiple objects per image <br>
4）More than 300,000 images <br>
5）More than 2 Million instances <br>
6）80 object categories <br>
7）5 captions per image <br>
8）Keypoints on 100,000 people<br>
COCO數據集由微軟贊助，其對於圖像的標註信息不僅有類別、位置信息，還有對圖像的語義文本描述，COCO數據集的開源使得近兩三年來圖像分割語義理解取得了巨大的進展，也幾乎成為了圖像語義理解算法性能評價的“標準”數據集。<br>
Google開源的開源了圖說生成模型show and tell就是在此數據集上測試的，想玩的可以下下來試試。數據集大小：~40GB <br>
[下載地址](http://mscoco.org/)

---

### **Day68** － Keras Sequential API：

難易度：:star::star:

了解 Keras Sequential API 與其應用的場景<br>
序列模型是多個網路層的線性堆疊<br>
Sequential 是一系列模型的簡單線性疊加，可以在構造函數中傳入一些列的網路層<br>

實用連結：<br>
[Getting started with the Keras Sequential model](https://keras.io/getting-started/sequential-model-guide/)

模型編譯：<br>
在訓練模型之前，您需要配置學習過程，的英文這通過compile方法完成的它接收三個參數：<br>
* 優化器optimizer。它可以是現有優化器的字符串標識符，如rmsprop或adagrad，也可以是Optimizer類的實例。
* 損失函數的損失，模型試圖最小化的目標函數它可以是現有損失函數的字符串標識符，如。categorical_crossentropy或mse，也可以是一個目標函數
* 評估標準指標。對於任何分類問題，你都希望將其設置為metrics = ['accuracy']。評估標準可以是現有的標準的字符串標識符，也可以是自定義的評估標準函數。

---

### **Day69** － Keras Module API：

難易度：:star::star:

了解 Keras Module API 與其應用的場景<br>

實用連結：<br>
[Getting started with the Keras functional API](https://keras.io/getting-started/functional-api-guide/)<br>
層「節點」的概念：<br>
每當你在某個輸入上調用一個層時，都將創建一個新的張量（層的輸出），並且為該層添加一個「節點」，將輸入張量連接到輸出張量。當多次調用同一個圖層時，該圖層將擁有多個節點索引(0, 1, 2...)。<br>
但是如果一個層與多個輸入連接呢？<br>
只要一個層僅僅連接到一個輸入，就不會有困惑，.output會返回層的唯一輸出<br>

全連接網路：<br>
Sequential 模型可能是實現這種網絡的一個更好選擇網路層的實例是可調用的，它以張量為參數，並且返回一個張量輸入和輸出均為張量，它們都可以用來定義一個模型（Model）<br>
這樣的模型同Keras的Sequential模型一樣，都可以被訓練

---

### **Day70** － Multi-layer Perception多層感知：

難易度：:star::star:

Multi-layer Perceptron (MLP)為一種監督式學習的演算法<br>
多層感知機是一種前向傳遞類神經網路，⾄至少包含三層結構(輸入層、隱藏層和輸出層)，並且利用到「倒傳遞」的技術達到學習(model learning)的監督式學習<br>

實用連結：<br>
機器學習 - 神經網路 (多層感知機 Multilayer perceptron, MLP)運作方式
[文章連結Medium](shorturl.at/oH234)<br>
[多層感知機](https://zh.wikipedia.org/wiki/%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E5%99%A8)

---

### **Day71** － 損失函數：

難易度：:star::star:

機器學習中所有的算法都需要最大化或最小化一個函數，這個函數被稱為「目標函數」。一般把最小化的一類函數，稱為「損失函數」，它能根據預測結果，衡量出模型預測能力的好壞<br>
損失函數大致可分為：分類問題的損失函數和回歸問題的損失函數<br>

實用連結：<br>
[TensorFlow筆記-06-神經網絡優化-​​損失函數，自定義損失函數，交叉熵](https://blog.csdn.net/qq_40147863/article/details/82015360)<br>
[使用損失函數](https://keras.io/losses/)<br>

---

### **Day72** － 啟動函數：

難易度：:star::star::star:<br>

了解啟動函數以及針對不同的問題使用合適的啟動函數<br>
啟動函數定義了每個節點（神經元）的輸出和輸入關係的函數為神經元提供規模化非線性化能力，讓神經網路路具備強大的擬合能力<br>

實用連結：<br>
在經典的人工神經網路解釋中，隱藏層中的所有神經元最初都是被啟動的，為了完成某一特定任務，有必要關閉其中的一些神經元，即有必要「遺忘」所有不必要信息。在人工神經網路中，啟動是指神經元在評估中參與正向傳播，在訓練中參與反向傳播。<br>
[神經網路常用啟動函數總結](https://zhuanlan.zhihu.com/p/39673127)<br>
[Reference 激活函數的圖示及其一階導數](https://dashee87.github.io/data%20science/deep%20learning/visualising-activation-functions-in-neural-networks/)<br>
[CS231N Lecture](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf)

---

### **Day73** － 梯度下降Gradient Descent：

難易度：:star::star::star:<br>

了解梯度下降(Gradient Descent)的定義<br>
機器學習算法當中，優化算法的功能，是通過改善訓練方式，來最小化(或最大化)損失函數，最常用的優化算法是梯度下降<br>

實用連結：<br>
[知乎 - Tensorflow中learning rate decay的技巧](https://zhuanlan.zhihu.com/p/32923584)<br>
[機器/深度學習-基礎數學(二):梯度下降法(gradient descent)](https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E6%95%B8%E5%AD%B8-%E4%BA%8C-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95-gradient-descent-406e1fd001f)

---

### **Day74** － Gradient Descent 數學原理：

難易度：:star::star::star:<br>

了解 Gradient Descent 的數學定義與程式樣貌<br>

實用連結：<br>
[機器學習-梯度下降法](shorturl.at/cgS49)<br>
[gradient descent using python and numpy](https://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy)<br>
[梯度下降算法的參數更新公式](https://blog.csdn.net/hrkxhll/article/details/80395033)

---

### **Day75** － BackPropagation：

難易度：:star::star::star:<br>

了解前行網路傳播(ForwardPropagation) /反向式傳播(BackPropagation )的差異<br>
了解反向式傳播BackPropagation的運作<br>

實用連結：<br>
[Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)<br>
[深度學習(Deep Learning)-反向傳播](https://ithelp.ithome.com.tw/articles/10198813)<br>
BP神經網路的原理及Python實現：<br>
[Blog連結](https://blog.csdn.net/conggova/article/details/77799464)<br>
[完整的結構化代碼連結](https://github.com/conggova/SimpleBPNetwork)

---

### **Day76** － 優化器optimizers：

難易度：:star::star::star:<br>

機器學習算法當中，大部分算法的本質就是建立優化模型，通過最優化方法對目標函數進行優化從而訓練出最好的模型<br>
優化算法的功能，是通過改善訓練方式，來最小化(或最大化)損失函數E(x)<br>
優化策略和算法，是用來更新和計算影響模型訓練和模型輸出的網絡參數，使其逼近或達到最優值<br>

實用連結：<br>
[An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf)<br>
在很多機器學習和深度學習的應用中，我們發現用的最多的優化器是Adam，為什麼呢？下面是TensorFlow中的優化器 [連結](https://www.tensorflow.org/api_guides/python/train)<br>
在keras中也有SGD，RMSprop，Adagrad，Adadelta，Adam等 [連結](https://keras.io/optimizers/)<br>
我們可以發現除了常見的梯度下降，還有Adadelta，Adagrad，RMSProp 等幾種優化器，都是什麼呢，又該怎麼選擇呢？ [連結](https://blog.csdn.net/qq_35860352/article/details/80772142)<br>
Sebastian Ruder的這篇論文中給出了常用優化器的比較 [連結](https://arxiv.org/pdf/1609.04747.pdf)

---

### **Day77** － 訓練神經網路的細節與技巧 - Validation and overfit：

難易度：:star::star:<br>

在練習當中檢視並了解 overfitting 現象<br>

實用連結：<br>
[Overfitting – Coursera 日誌](https://medium.com/@ken90242/machine-learning%E5%AD%B8%E7%BF%92%E6%97%A5%E8%A8%98-coursera%E7%AF%87-week-3-4-the-c05b8ba3b36f)<br>
[EliteDataScience – Overfitting](https://elitedatascience.com/overfitting-in-machine-learning)<br>
[Overfitting vs. Underfitting](https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765)

---

### **Day78** － 訓練神經網路前的注意事項：

難易度：:star:<br>

訓練模型前的檢查，訓練模型的時間跟成本都很大 <br>
開始訓練模型前應該檢查的環節：<br>
* 裝置
GPU
* 資料
* * 輸入是否正規化
* * 輸出是否正規化或獨熱編碼
* 模型架構
* 超參參數

實用連結：<br>
如何 Debugging？<br>
* 檢查程式碼<br>
* 養成好的程式撰寫習慣[(PEP8)](https://www.python.org/dev/peps/pep-0008/)
* 確認參數設定
* 欲實作的模型是否合適當前的資料
* 確認資料結構
* 資料是否足夠
* 是否乾淨
* 是否有適當的前處理
* 以簡單的方式實現想法
* 建立評估機制
* 開始循環測試 (evaluate - tuning - debugging)<br>
* 
[Troubleshooting Deep Neural Network – A Field Guide to Fix your Model](http://josh-tobin.com/assets/pdf/troubleshooting-deep-neural-networks-01-19.pdf)

---

### **Day79** － 訓練神經網路的細節與技巧 - Learning rate effect：

難易度：:star::star:<br>

了解 Learning Rate 對訓練的影響<br>
了解各優化器內，不同的參數對訓練的影響<br>

實用連結：<br>
* 知乎 - 深度學習超參數 Learning rate 與 Momentum 理解
* Learning rate：每次修正的幅度，太大則無法收斂，太小則修正過慢
* Weight decay：增加正則用以避免 overfitting
* Momentum：在修正方向上，增加動量，如牛頓定律一樣，增加動量有機會讓卡在局部最小值的狀態跳離
* Learning rate decay：讓Learning rate 可以隨訓練進行慢慢減小，讓收斂狀態趨於穩

[Estimating an Optimal Learning Rate For a Deep Neural Network](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0)<br>
[cs231n: learning and evaluation](http://cs231n.github.io/neural-networks-3/)<br>
[知乎-深度學習超參數簡單理解>>>>>>learning rate,weight decay 和 momentum](https://zhuanlan.zhihu.com/p/23906526)

---

### **Day80** － [練習 Day] 優化器與學習率的組合與比較：

難易度：:star::star::star:<br>

結合前面的知識與程式碼，比較不同的optimizer 與 learning rate 組合對訓練的結果與影響

---

### **Day81** － 訓練神經網路的細節與技巧 - Regularization：

難易度：:star::star::star:<br>

了解 regularization 的原理以及知道如何在 keras 中加入 regularization<br>
Regularizer 的效果：讓模型參數的數值較小 – 使得 Inputs 的改變不會讓 Outputs 有大幅的改變<br>

實用連結：<br>
[Toward Data Science-Regularization in Machine Learning](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)<br>
Machine Learning Explained:<br> [Regularization](http://enhancedatascience.com/2017/07/04/machine-learning-explained-regularization/)<br>
[機器學習：正規化 by Murphy](https://murphymind.blogspot.com/2017/05/machine.learning.regularization.html)<br>

---

### **Day82** － 訓練神經網路的細節與技巧 - Dropout：

難易度：:star::star:<br>

了解 dropout 的背景與可能可行的原理以及知道如何在 keras 中加入 dropout<br>
Dropout：在訓練時隨機將某些參數暫時設為 0 (刻意讓訓練難度提升)，強迫模型的每個參數有更強的泛化能力，也讓網路能在更多參數組合的狀態下習得表徵<br>

實用連結：<br>
[理解 Dropout – CSDN](https://blog.csdn.net/stdcoutzyx/article/details/49022443)<br>
[Dropout in Deep Learning](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5)

---

### **Day83** － 訓練神經網路的細節與技巧 - Batch normalization：

難易度：:star::star::star:<br>

理解 BatchNormalization 的原理以及知道如何在 keras 中加入 BatchNorm<br>

實用連結：<br>
[為何要使用 Batch Normalization – 莫煩 python](https://zhuanlan.zhihu.com/p/34879333)<br>
[Batch normalization 原理與實戰 – 知乎](https://zhuanlan.zhihu.com/p/34879333)

---

### **Day84** － [練習 Day] 正規化/機移除/批次標準化的 組合與比較：

難易度：:star::star::star:<br>

比較不同的regularization 的組合對訓練的結果與影響：如 dropout, regularizers, batch-normalization 等

---

### **Day85** － 訓練神經網路的細節與技巧 - 使用 callbacks 函數做 earlystop：

難易度：:star::star::star:<br>

了解如何在 Keras 中加入 callbacks以及知道 earlystop 的機制

實用連結：<br>
[Keras 的 EarlyStopping callbacks的使用與技巧 – CSND blog](https://blog.csdn.net/silent56_th/article/details/72845912)

---

### **Day86** － 訓練神經網路的細節與技巧 - 使用 callbacks 函數儲存 model：

難易度：:star::star:<br>

了解如何在訓練過程中，保留最佳的模型權重以及知道如何在 Keras 中，加入 ModelCheckPoint<br>
Model checkpoint：根據狀況隨時將模型存下來來，如此可以保證：
* 假如不幸訓練意外中斷，前面的功夫不會白費。我們可以從最近的一次繼續重新開始
* 我們可以透過監控 validation loss 來保證所存下來的模型是在 validation set 表現最好的一個

實用連結：<br>
[莫煩 Python - 儲存與載回模型](https://morvanzhou.github.io/tutorials/machine-learning/keras/3-1-save/)

---

### **Day87** － 訓練神經網路的細節與技巧 - 使用 callbacks 函數做 reduce learning rate：

難易度：:star::star:<br>

了解什麼是 Reduce Learning Rate以及知道如何在 Keras 中，加入 ReduceLearningRate

實用連結：<br>
Github 原碼：<br>
[LearningRateScheduler](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L910)<br>
與<br>
[ReduceLR](https://github.com/keras-team/keras/blob/master/keras/callbacks/callbacks.py#L946)

---

### **Day88** － 訓練神經網路的細節與技巧 - 撰寫自己的 callbacks 函數：

難易度：:star::star:<br>

學會如何使用自定義的 callbacks以及知道 callbacks 的啟動時機<br><br>
Callback 在訓練時的呼叫時機：<br>
* on_train_begin：在訓練最開始時
* on_train_end：在訓練結束時
* on_batch_begin：在每個 batch 開始時
* on_batch_end：在每個 batch 結束時
* on_epoch_begin：在每個 epoch 開始時
* on_epoch_end：在每個 epoch 結束時

實用連結：<br>
[Keras 中保留 f1-score 最高的模型 (per epoch)](https://zhuanlan.zhihu.com/p/51356820)

---

### **Day89** － 訓練神經網路的細節與技巧 - 撰寫自己的 Loss function：

難易度：:star::star:<br>

了解如何使用自定義的損失函數<br><br>
在 Keras 中，我們可以自行定義函式來進行損失的運算。一個損失函數必須：<br>
* 有 y_true 與 y_pred 兩個輸入
* 必須可以微分
* 必須使用 tensor operation，也就是在 tensor 的狀狀態下，進行運算

實用連結：<br>
[CSDN - Keras 自定義 Loss 函數](https://blog.csdn.net/A_a_ron/article/details/79050204)

---

### **Day90** － 使用傳統電腦視覺與機器學習進行影像辨識：

難易度：:star::star::star:<br>

了解用傳統電腦來做影像辨識的過程以及如何用顏色直方圖提取圖片的顏色特徵<br>

實用連結：<br>
[圖像分類 | 深度學習PK傳統機器學習](https://cloud.tencent.com/developer/article/1111702)<br>
[OpenCV - 直方圖](https://chtseng.wordpress.com/2016/12/05/opencv-histograms%E7%9B%B4%E6%96%B9%E5%9C%96/)<br>

進階參考資料： <br>
[OpenCV 教學文檔](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)<br>
[Udacity free course: Introduction To Computer Vision](https://www.udacity.com/course/introduction-to-computer-vision--ud810)

---

### **Day91** － [練習 Day] 使用傳統電腦視覺與機器學習進行影像辨識：

難易度：:star::star::star:<br>

體驗使用不同特徵來做機器學習分類問題的差別<br>
知道 hog 的調用方式<br>
知道 svm 在 opencv 的調用方式<br>

實用連結：<br>
[Sobel 運算子 wiki ](https://zh.wikipedia.org/wiki/%E7%B4%A2%E8%B2%9D%E7%88%BE%E7%AE%97%E5%AD%90)<br>
[基於傳統圖像處理的目標檢測與識別(HOG+SVM附代碼)](https://www.cnblogs.com/zyly/p/9651261.html)<br>
[知乎 - 什麼是 SVM](https://www.zhihu.com/question/21094489)<br>
[程式碼範例的來源，裡面用的是 mnist 來跑 ](https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/ml/py_svm_opencv/hogsvm.py)<br>
[範例來源裡使用的 digit.png 檔案位置](https://raw.githubusercontent.com/opencv/opencv/master/samples/data/digits.png)

---

### **Day92** － 卷積神經網路 (Convolution Neural Network, CNN) 簡介：

難易度：:star::star:<br>

了解甚麼是卷積神經網路 (Convolution Neural Network)<br>
了解卷積 (Convolution) 的定義與原理<br>
卷積神經網路目前在多數電腦視覺的任務中，都有優於人類的表現<br>
卷積是透過濾波器尋找圖型中特徵的一種數學運算<br>
卷積神經網路中的濾波器數值是從資料中自動學習出來的<br>

---

### **Day93** － 卷積神經網路架構細節：

難易度：:star::star::star::star:<br>

了解CNN 為何適用於 Image 處理<br>
了解卷積層中的卷積過程是如何計算的<br>

實用連結：<br>
[參考連結](http://matlabtricks.com/post-5/3x3-convolution-kernels-with-online-demo#demo)

---

### **Day94** － 卷積神經網路 - 卷積(Convolution)層與參數調整：

難易度：:star::star::star:<br>

了解CNN Flow以及卷積 (Convolution) 的超參數(Hyper parameter)設定與應用

---

### **Day95** － 卷積神經網路 - 池化(Pooling)層與參數調整：

難易度：:star::star::star:<br>

了解 CNN Flow以及池化層超參數的調適

---

### **Day96** － Keras 中的 CNN layers：

難易度：:star::star::star:<br>

了解 Keras 中的 CNN layers 如何使用了以及各項參數的意義

---

### **Day97** － 使用 CNN 完成 CIFAR-10 資料集：

難易度：:star::star::star:<br>

認識電腦視覺中著名的 Cifar-10 資料集<br>
了解如何使用 Keras 來完成 Cifar-10 的 CNN 分類模型<br>

---

### **Day98** － 訓練卷積神經網路的細節與技巧 - 處理大量數據：

難易度：:star::star::star:<br>

了解遇到資料量龐大的數據該如何處理<br>
了解如何撰寫 Python 的生成器 (generator) 程式碼<br>
當資料量太大無法一次讀進記憶體時，可使用 Generator 進行批次讀取<br>
使用 yield 來撰寫 Python generator

---

### **Day99** － 訓練卷積神經網路的細節與技巧 - 處理小量數據：

難易度：:star::star::star:<br>

了解遇到資料量較少的數據該如何處理<br>
了解資料增強的意義與使用時的注意事項<br>

---

### **Day100** － 訓練卷積神經網路的細節與技巧 - 轉移學習 (Transfer learning)：

難易度：:star::star::star:<br>

了解遷移學習 (Transfer learning) 的意義<br>
了解在 Keras 中如何使用遷移學習來訓練模型<br>

實用連結：<br>
[簡單使用 Keras 完成 transfer learning - 中文](https://ithelp.ithome.com.tw/articles/10190971)<br>
[Keras 作者教你用 pre-trained CNN 模型-英文](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb)

---

### **期末考Day_101~103_HW**：

難易度：:star::star::star::star:<br>

進行百日馬拉松期末考-貓狗圖像分類競賽：

貓與狗一直都是人類最重要的夥伴，無論你是狗派還是貓派，為自己訓練一個 CNN 模型來分辨貓狗吧！ 貓狗辨識之前是在 Kaggle 上熱門的競賽之一，非常適合剛學習完卷積神經網路的同學做練習。在本次的期末考，同學會需要應用這一百日中所學到的機器學習知識，來完成這項挑戰！

競賽連結：
[2nd_ML100Marathon_final](https://www.kaggle.com/c/ml-marathon-final)

---
