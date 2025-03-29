---
layout: model
title: English model_parse_xfm_bart_base_v0_1_0_pipeline pipeline BartTransformer from Iseratho
author: John Snow Labs
name: model_parse_xfm_bart_base_v0_1_0_pipeline
date: 2025-03-29
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

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`model_parse_xfm_bart_base_v0_1_0_pipeline` is a English model originally trained by Iseratho.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/model_parse_xfm_bart_base_v0_1_0_pipeline_en_5.5.1_3.0_1743224151453.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/model_parse_xfm_bart_base_v0_1_0_pipeline_en_5.5.1_3.0_1743224151453.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("model_parse_xfm_bart_base_v0_1_0_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("model_parse_xfm_bart_base_v0_1_0_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|model_parse_xfm_bart_base_v0_1_0_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|810.2 MB|

## References

https://huggingface.co/Iseratho/model_parse_xfm_bart_base-v0_1_0

## Included Models

- DocumentAssembler
- BartTransformer