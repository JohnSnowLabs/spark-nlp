---
layout: model
title: English e5_base_v2_mnli_anli BertForZeroShotClassification from mjwong
author: John Snow Labs
name: e5_base_v2_mnli_anli
date: 2024-08-31
tags: [en, open_source, onnx, zero_shot, bert]
task: Zero-Shot Classification
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForZeroShotClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForZeroShotClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`e5_base_v2_mnli_anli` is a English model originally trained by mjwong.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/e5_base_v2_mnli_anli_en_5.4.2_3.0_1725147323265.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/e5_base_v2_mnli_anli_en_5.4.2_3.0_1725147323265.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')
    
tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

zeroShotClassifier  = BertForZeroShotClassification.pretrained("e5_base_v2_mnli_anli","en") \
     .setInputCols(["document","token"]) \
     .setOutputCol("class")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, zeroShotClassifier])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")
    
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val zeroShotClassifier  = BertForZeroShotClassification.pretrained("e5_base_v2_mnli_anli", "en")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("class") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, zeroShotClassifier))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|e5_base_v2_mnli_anli|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|405.4 MB|

## References

https://huggingface.co/mjwong/e5-base-v2-mnli-anli