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
* **結語**

### :point_right: 我的[Kaggle](https://www.kaggle.com/kuoyuhong)

# 主題一：資料清理數據前處理

### **重點摘要：**
### **Day_005_HW** － EDA資料分佈：
### **Day_006_HW** － EDA: Outlier 及處理：
### **Day_008_HW** － DataFrame operationData frame merge/常用的 DataFrame 操作：
### **Day_011_HW** － EDA: 不同數值範圍間的特徵如何檢視/繪圖與樣式Kernel Density Estimation (KDE)：
### **Day_014_HW** － Subplots：
### **Day_015_HW** － Heatmap & Grid-plot：
### **Day_016_HW** － 模型初體驗 Logistic Regression：

## Outlier處理、數值標準化、離散化：

## 資料視覺化(Matplotlib、Seaborn)：

---

# 主題二：資料科學特徵工程技術

### **重點摘要：**
### **Day_019_HW** － 數值型特徵-補缺失值與標準化：
### **Day_020_HW** － 數值型特徵 - 去除離群值：
### **Day_022_HW** － 類別型特徵 - 基礎處理：
### **Day_023_HW** － 類別型特徵 - 均值編碼：
### **Day_024_HW** － 類別型特徵 - 其他進階處理：
### **Day_025_HW** － 時間型特徵：
### **Day_026_HW** － 特徵組合 - 數值與數值組合：
### **Day_027_HW** － 特徵組合 - 類別與數值組合：
### **Day_028_HW** － 特徵選擇：

## 類別型特徵處理：

### 標籤編碼(Label Encoder)：
類似於流水號，依序將新出現的類別依序編上新代碼，已出現的類別編上已使用的代碼<br>
確實能轉成分數，但缺點是分數的大小順序沒有意義<br>
![標籤編碼](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E6%A8%99%E7%B1%A4%E7%B7%A8%E7%A2%BC.png)

### 獨熱編碼(One Hot Encoder)：
為了改良數字大小沒有意義的問題，將不同的類別分別獨立為一欄<br>
缺點是需要較大的記憶空間與計算時間，且類別數量越多時越嚴重<br>
![獨熱編碼](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E7%8D%A8%E7%86%B1%E7%B7%A8%E7%A2%BC.png)
類別型特徵建議預設採用標籤編碼，除非該特徵重要性高，且可能值較少(獨熱編碼時負擔較低) 時，才應考慮使用獨熱編碼<br>
獨熱編碼缺點是需要較大的記憶空間與計算時間，且類別數量越多時越嚴重<br>

類別型特徵有標籤編碼 (Label Encoding) 與獨熱編碼(One Hot Encoding) 兩種基礎編碼方式<br>
兩種編碼中標籤編碼比較常用<br>
當特徵重要性高，且可能值較少時，才應該考慮獨熱編碼<br>

### 均值編碼(Mean Encoding)：
使用時機：類別特徵看起來來與目標值有顯著相關時，使用目標值的平均值，取代原本的類別型特徵<br>
![均值編碼](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E5%9D%87%E5%80%BC%E7%B7%A8%E7%A2%BC.png)
#### 平滑化：<br>
如果交易樣本非常少, 且剛好抽到極端值, 平均結果可能會有誤差很大<br>
因此, 均值編碼還需要考慮紀錄筆數, 當作可靠度的參考<br>
當平均值的可靠度低時, 我們會傾向相信全部的總平均<br>
當平均值的可靠度高時, 我們會傾向相信類別的平均<br>
依照紀錄筆數, 在這兩者間取折衷<br>

### 計數編碼(Counting)：
如果類別的目標均價與類別筆數呈正相關(或負相關)，也可以將筆數本身當成特徵例如 : 購物網站的消費金額預測<br>
![計數編碼](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E8%A8%88%E6%95%B8%E7%B7%A8%E7%A2%BC.png)

#### 方法：
```
# 加上 'Ticket' 欄位的計數編碼
# 第一行 : df.groupby(['Ticket']) 會輸出 df 以 'Ticket' 群聚後的結果, 但因為群聚一類只會有一個值, 因此必須要定義運算
# 例如 df.groupby(['Ticket']).size(), 但欄位名稱會變成 size, 要取別名就需要用語法 df.groupby(['Ticket']).agg({'Ticket_Count':'size'})
# 這樣出來的計數欄位名稱會叫做 'Ticket_Count', 因為這樣群聚起來的 'Ticket' 是 index, 所以需要 reset_index() 轉成一欄
# 因此第一行的欄位, 在第三行按照 'Ticket_Count' 排序後, 最後的 DataFrame 輸出如 Out[5]
count_df = df.groupby(['Ticket'])['Name'].agg({'Ticket_Count':'size'}).reset_index()
# 但是上面資料表結果只是 'Ticket' 名稱對應的次數, 要做計數編碼還需要第二行 : 將上表結果與原表格 merge, 合併於 'Ticket' 欄位
# 使用 how='left' 是完全保留原資料表的所有 index 與順序
df = pd.merge(df, count_df, on=['Ticket'], how='left')
count_df.sort_values(by=['Ticket_Count'], ascending=False).head(10)
```

### 特徵雜湊(Feature Hash)：
使用時機：相異類別的數量量非常龐大時，特徵雜湊是一種折衷方案<br>
將類別由雜湊函數定應到一組數字，調整雜湊函數對應值的數量<br>
在計算空間/時間與鑑別度間取折衷<br>
也提高了訊息密度，減少無用的標籤<br>
![特徵雜湊](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E7%89%B9%E5%BE%B5%E9%9B%9C%E6%B9%8A.png)

#### 方法：
```
# 這邊的雜湊編碼, 是直接將 'Ticket' 的名稱放入雜湊函數的輸出數值, 為了要確定是緊密(dense)特徵, 因此除以10後看餘數
# 這邊的 10 是隨機選擇, 不一定要用 10, 同學可以自由選擇夠小的數字試看看. 基本上效果都不會太好
df_temp['Ticket_Hash'] = df['Ticket'].map(lambda x:hash(x) % 10)
```

計數編碼是計算類別在資料中的出現次數，當目標平均值與類別筆數呈正/負相關時，可以考慮使用<br>
當相異類別數量相當大時，其他編碼方式效果更更差，可以考慮雜湊編碼以節省時間<br>
*註 : 雜湊編碼效果也不佳，這類問題更好的解法是嵌入式編碼(Embedding)<br>

### 時間型特徵：

時間也有週期的概念, 可以用週期合成一些重要的特徵聯聯想看看 : 有哪幾種時間週期, 可串聯到一些可做特徵的性質?<br>
年週期與春夏秋冬季節溫度相關<br>
月週期與薪水、繳費相關<br>
周週期與周休、消費習慣相關<br>
日週期與生理理時鐘相關<br>

前述的週期所需數值都可由時間欄位組成, 但還首尾相接<br>
因此週期特徵還需以正弦函數( sin )或餘弦函數( cos )加以組合<br>
例如 : 年週期 ( 正 : 冷 / 負 : 熱 )cos((⽉月/6 + ⽇日/180 )π)<br>
周週期 ( 正 : 精神飽滿/ 負 : 疲倦 )sin((星期幾/3.5 + ⼩小時/84 )π)<br>
日週期 ( 正 : 精神飽滿 / 負 : 疲倦 )sin((⼩小時/12 + 分/720 + 秒/43200 )π)<br>

時間型特徵最常用的是特徵分解 - 拆解成年/月/日/時/分/秒的分類值<br>
週期循環特徵是將時間"循環"特性改成特徵方式, 設計關鍵在於首尾相接, 因此我們需要使用 sin /cos 等週期函數轉換<br>
常見的週期循環特徵有 - 年週期(季節) / 周周期(例假日) / 日週期(日夜與生活作息), 要注意的是最高與最點的設置<br>

### 群聚編碼：
類似均值編碼的概念，可以取類別平均值 (Mean) 取代險種作為編碼<br>
但因為比較像性質描寫，因此還可以取其他統計值，如中位數 (Median)，眾數(Mode)，最大值(Max)，最小值(Min)，次數(Count)...等<br>
![群聚編碼](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E7%BE%A4%E8%81%9A%E7%B7%A8%E7%A2%BC.png)
![均值編碼&群聚編碼比較](https://github.com/KuoYuHong/2nd-ML100Days/blob/master/%E5%9C%96%E7%89%87/%E5%9D%87%E5%80%BC%E7%B7%A8%E7%A2%BC%26%E7%BE%A4%E8%81%9A%E7%B7%A8%E7%A2%BC%E6%AF%94%E8%BC%83.png)

---




