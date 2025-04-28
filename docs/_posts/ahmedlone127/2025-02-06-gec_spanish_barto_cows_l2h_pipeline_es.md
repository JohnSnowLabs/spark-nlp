---
layout: model
title: Castilian, Spanish gec_spanish_barto_cows_l2h_pipeline pipeline BartTransformer from SkitCon
author: John Snow Labs
name: gec_spanish_barto_cows_l2h_pipeline
date: 2025-02-06
tags: [es, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: es
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gec_spanish_barto_cows_l2h_pipeline` is a Castilian, Spanish model originally trained by SkitCon.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gec_spanish_barto_cows_l2h_pipeline_es_5.5.1_3.0_1738837182976.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gec_spanish_barto_cows_l2h_pipeline_es_5.5.1_3.0_1738837182976.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gec_spanish_barto_cows_l2h_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gec_spanish_barto_cows_l2h_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gec_spanish_barto_cows_l2h_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|810.3 MB|

## References

https://huggingface.co/SkitCon/gec-spanish-BARTO-COWS-L2H

## Included Models

- DocumentAssembler
- BartTransformer