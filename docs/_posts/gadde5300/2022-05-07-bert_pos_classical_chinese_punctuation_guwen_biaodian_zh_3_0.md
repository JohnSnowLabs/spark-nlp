---
layout: model
title: Chinese Part of Speech Tagger (from raynardj)
author: John Snow Labs
name: bert_pos_classical_chinese_punctuation_guwen_biaodian
date: 2022-05-07
tags: [bert, pos, part_of_speech, zh, open_source]
task: Part of Speech Tagging
language: zh
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Part of Speech model model, uploaded to Hugging Face, adapted and imported into Spark NLP. `classical-chinese-punctuation-guwen-biaodian` is a Chinese model orginally trained by `raynardj`.

## Predicted Entities

`】`, `？`, `，`, `；`, `【`, `（`, `！`, `）`, `。`, `"`, `：`, `'`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_pos_classical_chinese_punctuation_guwen_biaodian_zh_3.4.2_3.0_1651928580545.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_pos_classical_chinese_punctuation_guwen_biaodian_zh_3.4.2_3.0_1651928580545.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

    tokenClassifier = BertForTokenClassification.pretrained("bert_pos_classical_chinese_punctuation_guwen_biaodian","zh") \
        .setInputCols(["sentence", "token"]) \
        .setOutputCol("pos")

    pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

    data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

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

    val tokenClassifier = BertForTokenClassification.pretrained("bert_pos_classical_chinese_punctuation_guwen_biaodian","zh") 
        .setInputCols(Array("sentence", "token")) 
        .setOutputCol("pos")

    val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

    val data = Seq("I love Spark NLP").toDF("text")

    val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_pos_classical_chinese_punctuation_guwen_biaodian|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|zh|
|Size:|381.7 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/raynardj/classical-chinese-punctuation-guwen-biaodian
- https://github.com/raynardj/yuan
- https://github.com/raynardj/yuan
- https://github.com/raynardj/yuan