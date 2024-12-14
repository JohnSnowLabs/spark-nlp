---
layout: model
title: English deberta_base_long_nli_tasksource DeBertaForZeroShotClassification from tasksource
author: John Snow Labs
name: deberta_base_long_nli_tasksource
date: 2024-12-14
tags: [en, open_source, onnx, zero_shot, deberta]
task: Zero-Shot Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: DeBertaForZeroShotClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DeBertaForZeroShotClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`deberta_base_long_nli_tasksource` is a English model originally trained by tasksource.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_base_long_nli_tasksource_en_5.5.1_3.0_1734208001436.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_base_long_nli_tasksource_en_5.5.1_3.0_1734208001436.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

zeroShotClassifier  = DeBertaForZeroShotClassification.pretrained("deberta_base_long_nli_tasksource","en") \
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

val zeroShotClassifier  = DeBertaForZeroShotClassification.pretrained("deberta_base_long_nli_tasksource", "en")
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
|Model Name:|deberta_base_long_nli_tasksource|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|690.6 MB|

## References

https://huggingface.co/tasksource/deberta-base-long-nli