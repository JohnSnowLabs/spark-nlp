---
layout: model
title: Chinese BertForTokenClassification Tiny Cased model (from ckiplab)
author: John Snow Labs
name: bert_ner_bert_tiny_chinese_ner
date: 2022-08-02
tags: [bert, ner, open_source, zh]
task: Named Entity Recognition
language: zh
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-tiny-chinese-ner` is a Chinese model originally trained by `ckiplab`.

## Predicted Entities

`E-WORK_OF_ART`, `E-PRODUCT`, `S-PERCENT`, `E-EVENT`, `S-WORK_OF_ART`, `E-PERSON`, `MONEY`, `S-CARDINAL`, `E-LAW`, `PRODUCT`, `S-GPE`, `S-LANGUAGE`, `E-ORDINAL`, `S-MONEY`, `E-MONEY`, `QUANTITY`, `GPE`, `S-PERSON`, `EVENT`, `S-ORG`, `E-LOC`, `S-QUANTITY`, `PERCENT`, `E-TIME`, `CARDINAL`, `S-EVENT`, `NORP`, `S-LOC`, `WORK_OF_ART`, `E-PERCENT`, `DATE`, `S-PRODUCT`, `S-LAW`, `E-LANGUAGE`, `ORG`, `ORDINAL`, `FAC`, `TIME`, `LANGUAGE`, `LOC`, `E-NORP`, `E-QUANTITY`, `PERSON`, `E-GPE`, `E-ORG`, `S-ORDINAL`, `S-DATE`, `S-FAC`, `E-FAC`, `S-NORP`, `E-DATE`, `LAW`, `S-TIME`, `E-CARDINAL`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_bert_tiny_chinese_ner_zh_4.1.0_3.0_1659423788043.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_tiny_chinese_ner","zh") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("sentence"))
    .setOutputCol("token")
+
val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_tiny_chinese_ner","zh") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_bert_tiny_chinese_ner|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|zh|
|Size:|43.3 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/ckiplab/bert-tiny-chinese-ner
- https://github.com/ckiplab/ckip-transformers
- https://muyang.pro
- https://ckip.iis.sinica.edu.tw