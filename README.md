# Data_mining
兩題皆為Kaggle上的題目


1.Titanic
一開始先利用 matplotlib 去觀察每一個attribute間的關連性，以及一些相依性，然後爬討論區的文章，採納了裡面的一些建議去合併Data以及刪除Data，

最後用decision tree得到了0.78的準確率，

其中feature有 Pclass,Sex,Age,Alone,Fare,Embarked,Title,Age*class 後來為了得到更高的準確率開始觀察各個feature得到一些發現

(1)有cabin與沒cabin的人存活率差很多

(2)cabin開頭是B,C,D,E,F存活率遠高於50%

(3)Fare低的人存活率比較低

加入Cabin的feature準確率不會提升

把Fare調成3個區間15以下,15-60,60以上，準確率提升到0.808

用五個演算法來讓他們投票，高於三票的就存活，讓準確率提高到0.814

發現有些Title的平均年齡很低，將Title加入Age的預測是否可以提升準確率，結果準確率反而降低了，

處裡Age的區間，卻也無法提高準確率

將是否自己一個人搭船加入Age的預測但也無法使Age預測更準確

混合Age與Alone來參與預測卻也得不到好的結果

將每個演算法的權重用的不一樣，卻只能得到差不多的結果

最後以觀察每個test的data 加以修改才得到0.84的準確率。


2.Employee
一開始，我先觀察資料，發現沒有空白的資料，

利用one hot encode 跑 randomforest跟linear regression，發現跑出來的auc蠻高的，改了幾次參數之後，選擇了最高的auc的那一次model去預測結果，

但是上傳之後發現結果去意外的低，後來發現是overfitting的問題，做了10次cross validation，

雖然auc下降很多，但是再次上傳之後結果卻好很多，因此我得出了auc高不一定會比較好的結論0.88204再來，

我爬了一個kaggle上的文章，發現有一個greedy select feature的方法，重複的選擇較好的feature並更改其權重，

讓auc越來越高，但是缺點就是要跑很久很久，因為他的while loop跑得實在是很慢

最後得到了 0.89851。
