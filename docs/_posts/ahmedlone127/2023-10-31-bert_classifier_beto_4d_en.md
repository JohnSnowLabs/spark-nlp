---
layout: model
title: English BertForSequenceClassification Cased model (from ismaelardo)
author: John Snow Labs
name: bert_classifier_beto_4d
date: 2023-10-31
tags: [bert, sequence_classification, classification, open_source, en, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `BETO_4d` is a English model originally trained by `ismaelardo`.

## Predicted Entities

`9411`, `2433`, `8322`, `1323`, `5414`, `9412`, `2413`, `3343`, `1212`, `2522`, `9621`, `4321`, `2242`, `4225`, `7212`, `3331`, `5249`, `8344`, `2351`, `2431`, `3411`, `2411`, `1330`, `3322`, `1345`, `7127`, `8332`, `5223`, `5242`, `9333`, `2221`, `3511`, `4416`, `2141`, `3251`, `2161`, `4226`, `3344`, `5230`, `1324`, `3111`, `1219`, `3311`, `3257`, `2423`, `3512`, `2519`, `4323`, `9112`, `2143`, `2310`, `3321`, `5244`, `2635`, `4110`, `2421`, `7412`, `3118`, `5222`, `8343`, `1221`, `3122`, `2521`, `3115`, `2330`, `2529`, `3313`, `1211`, `3112`, `3611`, `2341`, `3113`, `2243`, `2513`, `8321`, `2342`, `3323`, `2145`, `2151`, `7233`, `2512`, `4214`, `3221`, `2424`, `2166`, `4222`, `3432`, `2642`, `2144`, `1412`, `2511`, `5120`, `9334`, `7231`, `4211`, `9321`, `2142`, `3142`, `2634`, `3312`, `3114`, `4311`, `1420`, `3334`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_beto_4d_en_5.1.4_3.4_1698790912100.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_beto_4d_en_5.1.4_3.4_1698790912100.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_beto_4d","en") \
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

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_beto_4d","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("en.classify.beto_bert").predict("""PUT YOUR STRING HERE""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_beto_4d|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|412.2 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

References

- https://huggingface.co/ismaelardo/BETO_4d