

# Dataset
## Human-annotated Data
The ungrammatical sentences are from the following public websites.
[1](https://wenku.baidu.com/view/1ce351635727a5e9846a610e?aggId=e4e228d30166f5335a8102d276a20029bc646366&fr=catalogMain_text_ernie_recall_v1%3Awk_recommend_main_graph&_wkts_=1686039387317&bdQuery=%E5%86%97%E4%BD%99%E7%97%85%E5%8F%A5%E7%BB%83%E4%B9%A0) [2](https://baijiahao.baidu.com/s?id=1675817725570818147&wfr=spider&for=pc) [3](https://easylearn.baidu.com/edu-page/tiangong/exercisedetail?id=174470eef8c75fbfc77db25d&from=search-duoti_pc-xiti_Detail_pc) [4](http://bj.xdf.cn/zhongkao/chuer/zhidao/134300.html) [5](http://bj.xdf.cn/zhongkao/chuer/zhidao/134299.html) [6](https://www.yueyeche.com.cn/zhjx/202207/19911.html) [7](https://mp.weixin.qq.com/s?__biz=MzI0NzE5NDI2MA==&mid=2652204429&idx=2&sn=6db3a396e1f1da2a56185917e8459d71&chksm=f2527a76c525f3600808e041222a6a78a49817314ad69603ab48129d31492a60b6920c8ac736&scene=27) [8](https://mp.weixin.qq.com/s?__biz=MzUzMDQ2MTM4OQ==&mid=2247557713&idx=4&sn=50caf0d739fd625a277e0d88fd97e1e8&chksm=fa52c5f3cd254ce57609af3da2a21e6fd0c7cdbb45d6a41cb3168c0e7e57b23b825508433d6e&scene=27) [9](https://wenku.baidu.com/view/5c9798cd961ea76e58fafab069dc5022aaea46f2.html?fr=aladdin664466&ind=3&_wkts_=1686039743632&bdQuery=%E5%8F%A5%E5%BC%8F%E6%9D%82%E7%B3%85) [10](https://zhuanlan.zhihu.com/p/479275444) [11](https://www.zszzs.com/wendang/qitafanwen/54091.html) [12](https://mp.weixin.qq.com/s?__biz=MzU4NTc3MzkwMw==&mid=2247500319&idx=3&sn=6ba362341e8f5543a8bb815e3a1657bd&chksm=fd87e43fcaf06d29a7486e45fa98215710987154fe9fcd58df33a4abf676699be2d44c293646&scene=27) [13](https://baijiahao.baidu.com/s?id=1742587369710610978&wfr=spider&for=pc) [14](https://mp.weixin.qq.com/s/DQnlXE_bKrSmTUVqTesqIg) [15](https://baijiahao.baidu.com/s?id=1617092703098480309&wfr=spider&for=pc) [16](https://www.renrendoc.com/paper/208183328.html)

# ChatGPT-generated Data
Using ChatGPT to generate ungrammatical sentences according to the [Clues](https://wenku.baidu.com/view/1ce351635727a5e9846a610e?aggId=e4e228d30166f5335a8102d276a20029bc646366&fr=catalogMain_text_ernie_recall_v1%3Awk_recommend_main_graph&_wkts_=1686039387317&bdQuery=%E5%86%97%E4%BD%99%E7%97%85%E5%8F%A5%E7%BB%83%E4%B9%A0)
![](ChatGPT.png)


  # Training
```
python finetuning.py
```
# Inferencing
```
python generate.py
```


