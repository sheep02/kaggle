# Freesound General-Purpose Audio Tagging Challenge

　およそ9000件の教師データを用いて分類する.ラベルは41種類.

### FIRST TRIAL

　フーリエ変換を行いそのスペクトルの画像を64x64へリサイズしResnet18/BatchNormalizationにて学習.波形が激しい区間に
多くの情報量が含まれていると仮定したがこの最大分散区間で抽出したところ訓練精度は上昇したものの汎化性能が低下してしまった.   

　mixupなどの汎化テクニックや転移学習を知っていながら思い出せずに利用し損ねた.



### SECOND TRIAL

　時間軸に依存しない特徴抽出を目的としてwavファイルを一定区間で分割, ハミング窓を適用, ケプストラムの低次元情報は話者識別に利用されているとのことからこれを対象として抽出を行った. 
　K-meansによるクラスタリングを行いデータの圧縮を試みた,がしかし,結果は芳しくないものだった.   

　話者識別に利用されているということは声帯の特徴が顕著であるということ, それは進化の過程で可聴域とともに環境音と帯域を異にするようになっている可能性があることを考慮すれば短慮であったと言わざるを得ない.

  

# LANL Earthquake Prediction

### FIRST TRIAL

　特徴を示すタイムスタンプの間隔の裏に潜む機材の推察を行い, 有効な特徴量を見出そうと試みた.

  

　可視化に執心してしまいあたかも規則性があるかのような演出をしてしまった. 妄想しすぎずに検定で検証できる程度の仮定でデータに忠実であることが大切だ. 泣きたくなってきた.

### SECOND TRIAL

　ヒルベルト変換後の特徴量やハースト指数を用いたがスコアは伸びずじまいであった. 遺伝的アルゴリズムによる特徴量生成, optunaによるハイパーパラメータの最適化を実行.



  

# Predicting Molecular Properties

　様々な特徴量算出を容易にする目的で分子座標の規格化を行った. 二分子間の中点からその線分上に単位ベクトルを設定, これを法線として平面の方程式から任意の単位ベクトルを設定, 外積から直交する単位ベクトルを設定. これらを新たな基底として基底変換を行い絶対座標から相対座標を元にした幾何的な特徴量の算出を容易にした.  

　特徴量が量子的な性質を示すらしい. よってsigmoid関数を加算し階段状にしたQuantized Sigmoidはどうかと考えた. しかしそんなことをせずともNNは関数の局所構造に強いらしい. (Safran (2017), Imaizumi&Fukumizu (2019))

  

　かくかくシカジカ, 0サブである.


