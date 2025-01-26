---
layout: model
title: English test2_hdvd2309_pipeline pipeline GPT2Transformer from hdvd2309
author: John Snow Labs
name: test2_hdvd2309_pipeline
date: 2025-01-25
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

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`test2_hdvd2309_pipeline` is a English model originally trained by hdvd2309.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/test2_hdvd2309_pipeline_en_5.5.1_3.0_1737825013210.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/test2_hdvd2309_pipeline_en_5.5.1_3.0_1737825013210.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("test2_hdvd2309_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("test2_hdvd2309_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|test2_hdvd2309_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|73.7 MB|

## References

https://huggingface.co/hdvd2309/test2

## Included Models

- DocumentAssembler
- GPT2Transformer