# 概要
整数計画問題の一つである巡回セールスマン問題(TSP)をPythonのライブラリの一つであるPulpを用いてモデリングする．  
ソルバーとしてはpulpにデフォルトでついているCBCを使用している.  
実際，頂点数が30を超えたあたりから実行時間が長くなっていくのでCplex等の有償ソルバーに変更する必要が出てくる.  
Pulpでのソルバーの変更方法については[こちら](http://inarizuuuushi.hatenablog.com/entry/2019/03/07/090000)の記事を参考にする.  
またlpファイルを吐き出させてlpからsolverに投げる方法もある.  

# 最適解の可視化  
定式して得られた答えからnetworkxを利用して経路を可視化したもの.  
## N=10
![N10](/Picture/N10.png)
## N=20
![N20](/Picture/N20.png)
