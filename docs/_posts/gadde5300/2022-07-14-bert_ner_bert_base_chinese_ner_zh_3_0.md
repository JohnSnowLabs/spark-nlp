---
layout: model
title: Chinese BertForTokenClassification Base Cased model (from ckiplab)
author: John Snow Labs
name: bert_ner_bert_base_chinese_ner
date: 2022-07-14
tags: [bert, ner, open_source, zh]
task: Named Entity Recognition
language: zh
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-chinese-ner` is a Chinese model originally trained by `ckiplab`.

## Predicted Entities

`TIME`, `E-PRODUCT`, `S-QUANTITY`, `PRODUCT`, `S-PRODUCT`, `CARDINAL`, `EVENT`, `LAW`, `E-CARDINAL`, `PERSON`, `S-DATE`, `S-LAW`, `S-FAC`, `E-PERCENT`, `FAC`, `WORK_OF_ART`, `E-ORDINAL`, `S-GPE`, `LANGUAGE`, `S-ORDINAL`, `S-ORG`, `QUANTITY`, `ORDINAL`, `ORG`, `E-TIME`, `E-LOC`, `PERCENT`, `E-LANGUAGE`, `E-LAW`, `E-NORP`, `E-PERSON`, `S-CARDINAL`, `S-LANGUAGE`, `S-PERSON`, `E-WORK_OF_ART`, `S-LOC`, `S-TIME`, `E-FAC`, `S-NORP`, `S-WORK_OF_ART`, `E-GPE`, `LOC`, `E-DATE`, `S-EVENT`, `NORP`, `E-QUANTITY`, `S-MONEY`, `DATE`, `E-EVENT`, `E-ORG`, `GPE`, `S-PERCENT`, `E-MONEY`, `MONEY`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_bert_base_chinese_ner_zh_4.0.0_3.0_1657805016310.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_ner_bert_base_chinese_ner_zh_4.0.0_3.0_1657805016310.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_base_chinese_ner","zh") \
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

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_base_chinese_ner","zh") 
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
|Model Name:|bert_ner_bert_base_chinese_ner|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|zh|
|Size:|381.7 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/ckiplab/bert-base-chinese-ner
- https://github.com/ckiplab/ckip-transformers
- https://ckip.iis.sinica.edu.tw
- https://muyang.pro