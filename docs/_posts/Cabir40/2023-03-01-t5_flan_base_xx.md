---
layout: model
title: English T5ForConditionalGeneration Cased model (from google)
author: John Snow Labs
name: t5_flan_base
date: 2023-03-01
tags: [open_source, t5, flan, xx, tensorflow]
task: Text Generation
language: xx
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: T5Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. flan-t5-base is a English model originally trained by google.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_flan_base_xx_4.3.0_3.0_1677702524850.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_flan_base_xx_4.3.0_3.0_1677702524850.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
.setInputCols("text") \
.setOutputCols("document")

t5 = T5Transformer.pretrained("t5_flan_base","xx") \
.setInputCols("document") \
.setOutputCol("answers")

pipeline = Pipeline(stages=[documentAssembler, t5])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text")
.setOutputCol("document")

val t5 = T5Transformer.pretrained("t5_flan_base","xx") 
.setInputCols("document")
.setOutputCol("answers")

val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_flan_base|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[t5]|
|Language:|xx|
|Size:|1.0 GB|

## References

https://huggingface.co/google/flan-t5-base
