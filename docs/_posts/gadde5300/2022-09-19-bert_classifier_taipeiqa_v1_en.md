---
layout: model
title: English BertForSequenceClassification Cased model (from nicktien)
author: John Snow Labs
name: bert_classifier_taipeiqa_v1
date: 2022-09-19
tags: [bert, sequence_classification, classification, open_source, en]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `TaipeiQA_v1` is a English model originally trained by `nicktien`.

## Predicted Entities

`臺北市就業服務處`, `臺北市停車管理工程處`, `臺北市政府教育局終身教育科`, `臺北市政府教育局國小教育科`, `臺北市政府衛生局醫事管理科`, `臺北市商業處`, `臺北市政府地政局土地開發科`, `臺北市信義區公所`, `臺北市立天文科學教育館`, `臺北市市場處公有零售市場科`, `臺北市政府地政局地權及不動產交易科`, `臺北市藝文推廣處`, `臺北市政府環境保護局北投垃圾焚化廠`, `臺北市政府產業發展局科技產業服務中心`, `臺北市政府警察局刑事警察大隊`, `臺北市政府資訊局應用服務組`, `臺北市動物保護處動物救援隊`, `臺北市交通事件裁決所違規申訴課`, `臺北市中山堂管理所`, `臺北市立文獻館`, `臺北市勞動檢查處`, `臺北市政府勞動局`, `臺北市政府資訊局綜合企劃組`, `臺北市政府都市發展局`, `臺北市立聯合醫院昆明防治中心`, `臺北市立美術館`, `臺北市青少年發展處`, `臺北市政府環境保護局環保稽查大隊`, `臺北市立聯合醫院`, `臺北市政府環境保護局環境清潔管理科`, `臺北市政府捷運工程局南區工程處`, `臺北市政府交通局交通治理科`, `臺北市政府工務局新建工程處工程配合科`, `臺北市政府客家事務委員會`, `臺北市政府工務局大地工程處`, `臺北市建築管理工程處使用科`, `臺北市內湖區公所`, `臺北市市政大樓公共事務管理中心`, `臺北市政府社會局`, `臺北市公共運輸處大眾運輸科`, `臺北市交通管制工程處設施科`, `臺北市政府人事處考訓科`, `臺北市都市更新處`, `臺北市交通事件裁決所案件管理課`, `臺北市政府交通局`, `臺北自來水事業處`, `臺北市政府環境保護局秘書室`, `臺北市公共運輸處業務稽查科`, `臺北市建築管理工程處公寓大廈科`, `臺北市稅捐稽徵處稅務管理科`, `臺北市政府勞動局職業安全衛生科`, `臺北市動物保護處產業保育組`, `臺北市政府教育局綜合企劃科`, `臺北市政府工務局衛生下水道工程處營運管理科`, `臺北市政府民政局戶籍行政科`, `臺北市政府勞動局勞資關係科`, `臺北市政府捷運工程局機電系統工程處`, `臺北市政府環境保護局廢棄物處理場`, `臺北市政府研究發展考核委員會`, `臺北市稅捐稽徵處法務科`, `臺北市政府工務局公園路燈工程管理處青年公園管理所`, `臺北市動物保護處防疫檢驗組`, `臺北市政府教育局特殊教育科`, `臺北市政府衛生局長期照護科`, `臺北翡翠水庫管理局`, `臺北市政府衛生局食品藥物管理科`, `臺北市政府民政局區政監督科`, `臺北市政府教育局中等教育科`, `臺北市交通事件裁決所違規裁罰課`, `臺北市勞動力重建運用處`, `臺北市建築管理工程處施工科`, `臺北市動物保護處動物管理組`, `臺北市政府社會局婦女福利及兒童托育科`, `臺北市政府環境保護局木柵垃圾焚化廠`, `臺北市政府消防局`, `臺北市政府衛生局疾病管制科`, `臺北市政府主計處`, `臺北市政府產業發展局工商服務科`, `臺北市政府社會局社會工作科`, `臺北市政府社會局兒童及少年福利科`, `臺北市立陽明教養院`, `臺北市政府人事處給與科`, `臺北市政府體育局`, `臺北市政府工務局`, `臺北市政府警察局交通警察大隊`, `臺北市政府工務局公園路燈工程管理處園藝工程隊`, `臺北市公共運輸處一般運輸科`, `臺北市政府原住民族事務委員會`, `臺北市政府環境保護局內湖垃圾焚化廠`, `臺北市政府勞動局勞動教育文化科`, `臺北市建築管理工程處建照科`, `臺北市立圖書館`, `臺北市政府地政局測繪科`, `臺北市政府警察局`, `臺北市稅捐稽徵處企劃服務科`, `臺北市動產質借處業務組`, `臺北市政府工務局新建工程處共同管道科`, `臺北市政府文化局藝術發展科`, `臺北市政府財政局菸酒暨稅務管理科`, `臺北市政府環境保護局資源循環管理科`, `臺北市政府工務局衛生下水道工程處迪化污水處理廠`, `臺北市政府警察局婦幼警察隊`, `臺北市職能發展學院`, `臺北市政府衛生局健康管理科`, `臺北市政府教育局人事室`, `臺北市政府教育局體育及衛生保健科`, `臺北市政府工務局公園路燈工程管理處`, `臺北市政府工務局水利工程處河川管理科`, `臺北市政府民政局人口政策科`, `臺北市政府社會局老人福利科`, `臺北市公共運輸處綜合規劃科`, `臺北市政府觀光傳播局`, `臺北市政府勞動局勞動基準科`, `臺北市政府都市發展局都市測量科`, `臺北市政府教育局學前教育科`, `臺北市稅捐稽徵處機會稅科`, `臺北市政府環境保護局廢棄物處理管理科`, `臺北市政府文化局`, `臺北市政府兵役局`, `臺北市政府環境保護局綜合企劃科`, `臺北市政府捷運工程局`, `臺北市政府社會局人民團體科`, `臺北市政府民政局宗教禮俗科`, `臺北市政府環境保護局空污噪音防制科`, `臺北市政府地政局地用科`, `臺北市動物保護處動物收容組`, `臺北市立動物園`, `臺北市政府環境保護局水質病媒管制科`, `臺北市政府法務局`, `臺北市政府資訊局設備網路組`, `臺北市政府教育局軍訓室`, `臺北大眾捷運股份有限公司`, `臺北市政府地政局地價科`, `臺北市政府社會局身心障礙者福利科`, `臺北市市場處批發市場科`, `臺北市稅捐稽徵處財產稅科`, `臺北市政府工務局衛生下水道工程處`, `臺北市政府勞動局就業安全科`, `臺北市市場處攤販科`, `臺北市政府地政局土地登記科`, `臺北市政府秘書處`, `臺北市政府教育局資訊教育科`, `臺北市政府社會局社會救助科`, `臺北市政府公務人員訓練處`, `臺北市政府產業發展局公用事業科`, `臺北市政府產業發展局農業發展科`, `臺北市交通事件裁決所肇事鑑定課`, `臺北市殯葬管理處`, `臺北市政府工務局水利工程處雨水下水道工程科`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_taipeiqa_v1_en_4.1.0_3.0_1663607358239.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_taipeiqa_v1_en_4.1.0_3.0_1663607358239.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_taipeiqa_v1","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_taipeiqa_v1","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_taipeiqa_v1|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|384.2 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/nicktien/TaipeiQA_v1