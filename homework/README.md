# 各主題精華重點說明(待更新)

* **1-資料清理數據前處理**
* **2-資料科學特徵工程技術**
* **3-機器學習基礎模型建立**
* **4-機器學習調整參數**
* **期中考-Kaggle競賽**
* **5-非監督式機器學習**
* **6-深度學習理論與實作**
* **7-初探深度學習使用Keras**
* **8-深度學習應用卷積神經網路**
* **期末考-Kaggle競賽**
* **結論**

### :point_right: 我的[Kaggle](https://www.kaggle.com/kuoyuhong)

## 主題一：資料清理數據前處理

### **Day_005_HW** － EDA資料分佈：
### **Day_006_HW** － EDA: Outlier 及處理：
### **Day_008_HW** － DataFrame operationData frame merge/常用的 DataFrame 操作：
### **Day_011_HW** － EDA: 不同數值範圍間的特徵如何檢視/繪圖與樣式Kernel Density Estimation (KDE)：
### **Day_014_HW** － Subplots：
### **Day_015_HW** － Heatmap & Grid-plot：
### **Day_016_HW** － 模型初體驗 Logistic Regression：

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

### **Day_011_HW** (2019-04-26) － EDA: 不同數值範圍間的特徵如何檢視/繪圖與樣式Kernel Density Estimation (KDE)：

難易度：:star::star::star:

針對年齡分組與排序，**畫出KDE圖**和**長條圖**<br>
在**使用seaborn套件畫KDE圖**時花較多時間摸索

```
label = 'target == 1'：在圖表中顯示說明的圖例

Seaborn方法：(matplotlib基礎)
對於長條圖而言，Seaborn有 distplot() 方法，可以將單變數分佈的長條圖和kde同時繪製出來
```
---

### **Day_014_HW** (2019-04-29) － Subplots：

難易度：:star::star:

使用**subplot**在不同位置畫圖形出來，同樣也是要**觀察資料、清理NA值**<br>
有許多好用的畫圖方法，需要來好好研究

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

---

## 主題二：資料科學特徵工程技術

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


---

### **Day_028_HW** (2019-05-15) － 特徵選擇：


---





