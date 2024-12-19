---
layout: model
title: English distilbart_xsum_6_6_sshleifer_pipeline pipeline BartTransformer from sshleifer
author: John Snow Labs
name: distilbart_xsum_6_6_sshleifer_pipeline
date: 2024-12-19
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbart_xsum_6_6_sshleifer_pipeline` is a English model originally trained by sshleifer.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbart_xsum_6_6_sshleifer_pipeline_en_5.5.1_3.0_1734580273912.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbart_xsum_6_6_sshleifer_pipeline_en_5.5.1_3.0_1734580273912.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilbart_xsum_6_6_sshleifer_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilbart_xsum_6_6_sshleifer_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbart_xsum_6_6_sshleifer_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|796.1 MB|

## References

https://huggingface.co/sshleifer/distilbart-xsum-6-6

## Included Models

- DocumentAssembler
- BartTransformer