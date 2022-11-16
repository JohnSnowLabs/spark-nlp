---
layout: model
title: English BertForTokenClassification Base Cased model (from QCRI)
author: John Snow Labs
name: bert_ner_bert_base_multilingual_cased_sem_english
date: 2022-07-14
tags: [bert, ner, open_source, en]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-multilingual-cased-sem-english` is a English model originally trained by `QCRI`.

## Predicted Entities

`ALT`, `CON`, `ENT`, `EXN`, `MOR`, `RLI`, `EMP`, `ROL`, `DEF`, `FUT`, `DOM`, `EXS`, `UNK`, `UOM`, `EQA`, `EPG`, `EXG`, `ART`, `LES`, `NAT`, `DEC`, `EPT`, `QUE`, `TOP`, `MOY`, `NEC`, `QUA`, `PRO`, `PST`, `DIS`, `COO`, `DST`, `IMP`, `ORG`, `REF`, `COM`, `SUB`, `PER`, `ETV`, `EPS`, `EXC`, `DOW`, `APP`, `INT`, `PRX`, `BUT`, `NOT`, `EXT`, `NOW`, `POS`, `LOC`, `AND`, `HAS`, `EFS`, `ENS`, `REL`, `NIL`, `HAP`, `YOC`, `IST`, `GPE`, `ITJ`, `SCO`, `EXV`, `ENG`, `ETG`, `TIM`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_bert_base_multilingual_cased_sem_english_en_4.0.0_3.0_1657805810031.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_base_multilingual_cased_sem_english","en") \
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

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_base_multilingual_cased_sem_english","en") 
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
|Model Name:|bert_ner_bert_base_multilingual_cased_sem_english|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|665.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/QCRI/bert-base-multilingual-cased-sem-english