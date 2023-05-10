---
layout: model
title: Persian Named Entity Recognition (from HooshvareLab)
author: John Snow Labs
name: bert_ner_bert_fa_zwnj_base_ner
date: 2022-05-09
tags: [bert, ner, token_classification, fa, open_source]
task: Named Entity Recognition
language: fa
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, uploaded to Hugging Face, adapted and imported into Spark NLP. `bert-fa-zwnj-base-ner` is a Persian model orginally trained by `HooshvareLab`.

## Predicted Entities

`LOC`, `PRO`, `MON`, `TIM`, `PER`, `DAT`, `FAC`, `EVE`, `PCT`, `ORG`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_bert_fa_zwnj_base_ner_fa_3.4.2_3.0_1652099703343.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_ner_bert_fa_zwnj_base_ner_fa_3.4.2_3.0_1652099703343.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_fa_zwnj_base_ner","fa") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["من عاشق جرقه nlp هستم"]]).toDF("text")

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

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_fa_zwnj_base_ner","fa") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("من عاشق جرقه nlp هستم").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_bert_fa_zwnj_base_ner|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|fa|
|Size:|442.2 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/HooshvareLab/bert-fa-zwnj-base-ner
- https://github.com/HaniehP/PersianNER
- http://nsurl.org/2019-2/tasks/task-7-named-entity-recognition-ner-for-farsi/
- https://elisa-ie.github.io/wikiann/
- https://github.com/hooshvare/parsner/issues