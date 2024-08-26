---
layout: model
title: Arabic msmarco_arabic_mt5_base_v1_pipeline pipeline T5Transformer from doc2query
author: John Snow Labs
name: msmarco_arabic_mt5_base_v1_pipeline
date: 2024-08-26
tags: [ar, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ar
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`msmarco_arabic_mt5_base_v1_pipeline` is a Arabic model originally trained by doc2query.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/msmarco_arabic_mt5_base_v1_pipeline_ar_5.4.2_3.0_1724634745759.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/msmarco_arabic_mt5_base_v1_pipeline_ar_5.4.2_3.0_1724634745759.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("msmarco_arabic_mt5_base_v1_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("msmarco_arabic_mt5_base_v1_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|msmarco_arabic_mt5_base_v1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|2.5 GB|

## References

https://huggingface.co/doc2query/msmarco-arabic-mt5-base-v1

## Included Models

- DocumentAssembler
- T5Transformer