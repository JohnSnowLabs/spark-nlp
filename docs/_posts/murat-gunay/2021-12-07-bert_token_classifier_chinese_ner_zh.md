---
layout: model
title: Chinese NER Model
author: John Snow Labs
name: bert_token_classifier_chinese_ner
date: 2021-12-07
tags: [chinese, token_classifier, bert, zh, open_source]
task: Named Entity Recognition
language: zh
edition: Spark NLP 3.3.2
spark_version: 2.4
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` and it's been fine-tuned for traditional Chinese language, leveraging `Bert` embeddings and `BertForTokenClassification` for NER purposes.

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_chinese_ner_zh_3.3.2_2.4_1638881767667.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_chinese_ner_zh_3.3.2_2.4_1638881767667.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_chinese_ner", "zh"))\
  .setInputCols(["sentence",'token'])\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")
      
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)
text = """我是莎拉，我从 1999 年 11 月 2 日。开始在斯图加特的梅赛德斯-奔驰公司工作。"""
result = model.transform(spark.createDataFrame([[text]]).toDF("text"))
```
```scala
val documentAssembler = DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_chinese_ner", "zh"))\
  .setInputCols(Array("sentence","token"))\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(Array("sentence", "token", "ner"))\
      .setOutputCol("ner_chunk")
      
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["我是莎拉，我从 1999 年 11 月 2 日。开始在斯图加特的梅赛德斯-奔驰公司工作。"].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("zh.ner.bert_token").predict("""我是莎拉，我从 1999 年 11 月 2 日。开始在斯图加特的梅赛德斯-奔驰公司工作。""")
```

</div>

## Results

```bash
+-----------------+---------+
|chunk            |ner_label|
+-----------------+---------+
|莎拉             |PERSON   |
|1999 年 11 月 2  |DATE     |
|斯图加特          |GPE      |
|梅赛德斯-奔驰公司  |ORG      |
+-----------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_chinese_ner|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|zh|
|Case sensitive:|true|
|Max sentense length:|256|

## Data Source

[https://huggingface.co/ckiplab/bert-base-chinese-ner](https://huggingface.co/ckiplab/bert-base-chinese-ner)

## Benchmarking

```bash
label   score
   f1   0.8118
```
