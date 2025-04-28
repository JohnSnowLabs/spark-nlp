---
layout: model
title: Korean abelism_miner_ver_3_pipeline pipeline BertForSequenceClassification from NeinYeop
author: John Snow Labs
name: abelism_miner_ver_3_pipeline
date: 2025-04-09
tags: [ko, open_source, pipeline, onnx]
task: Text Classification
language: ko
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`abelism_miner_ver_3_pipeline` is a Korean model originally trained by NeinYeop.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/abelism_miner_ver_3_pipeline_ko_5.5.1_3.0_1744180253076.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/abelism_miner_ver_3_pipeline_ko_5.5.1_3.0_1744180253076.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("abelism_miner_ver_3_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("abelism_miner_ver_3_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|abelism_miner_ver_3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|408.5 MB|

## References

https://huggingface.co/NeinYeop/abelism-miner_ver.3

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification