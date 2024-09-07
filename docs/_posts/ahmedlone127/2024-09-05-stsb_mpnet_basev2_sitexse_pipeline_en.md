---
layout: model
title: English stsb_mpnet_basev2_sitexse_pipeline pipeline MPNetForSequenceClassification from Kigo1974
author: John Snow Labs
name: stsb_mpnet_basev2_sitexse_pipeline
date: 2024-09-05
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`stsb_mpnet_basev2_sitexse_pipeline` is a English model originally trained by Kigo1974.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stsb_mpnet_basev2_sitexse_pipeline_en_5.5.0_3.0_1725575009193.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/stsb_mpnet_basev2_sitexse_pipeline_en_5.5.0_3.0_1725575009193.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("stsb_mpnet_basev2_sitexse_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("stsb_mpnet_basev2_sitexse_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stsb_mpnet_basev2_sitexse_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|408.2 MB|

## References

https://huggingface.co/Kigo1974/stsb-mpnet-basev2-sitexse

## Included Models

- DocumentAssembler
- TokenizerModel
- MPNetForSequenceClassification