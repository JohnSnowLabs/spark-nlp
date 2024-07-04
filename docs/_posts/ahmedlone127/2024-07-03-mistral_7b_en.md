---
layout: model
title: Mistral text-to-text model 7b int8
author: John Snow Labs
name: mistral_7b
date: 2024-07-03
tags: [mistral, en, llm, open_source, openvino]
task: Text Generation
language: en
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
engine: openvino
annotator: MistralTransformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MistralTransformer, adapted and imported into Spark NLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mistral_7b_en_5.4.0_3.0_1720021606199.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mistral_7b_en_5.4.0_3.0_1720021606199.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

documentAssembler = DocumentAssembler() \
	.setInputCol('text') \
	.setOutputCol('document')

mistral = MistralTransformer .pretrained() \
	.setMaxOutputLength(50) \
	.setDoSample(False) \
	.setInputCols(["document"]) \
	.setOutputCol("mistral_generation")

pipeline = Pipeline().setStages([documentAssembler, mistral])
data = spark.createDataFrame([["Who is the founder of Spark-NLP?"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
	.setInputCols("text")
	.setOutputCols("document")

val mistral = MistralTransformer .pretrained()
	.setMaxOutputLength(50)
	.setDoSample(False)
	.setInputCols(["document"])
	.setOutputCol("mistral_generation")

val pipeline = new Pipeline().setStages(Array(documentAssembler, mistral))
val data = Seq("Who is the founder of Spark-NLP?").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mistral_7b|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|6.6 GB|
