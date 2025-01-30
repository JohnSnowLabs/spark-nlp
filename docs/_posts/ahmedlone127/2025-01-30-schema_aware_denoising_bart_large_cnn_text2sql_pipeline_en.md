---
layout: model
title: English schema_aware_denoising_bart_large_cnn_text2sql_pipeline pipeline BartTransformer from shahrukhx01
author: John Snow Labs
name: schema_aware_denoising_bart_large_cnn_text2sql_pipeline
date: 2025-01-30
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

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`schema_aware_denoising_bart_large_cnn_text2sql_pipeline` is a English model originally trained by shahrukhx01.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/schema_aware_denoising_bart_large_cnn_text2sql_pipeline_en_5.5.1_3.0_1738240277774.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/schema_aware_denoising_bart_large_cnn_text2sql_pipeline_en_5.5.1_3.0_1738240277774.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("schema_aware_denoising_bart_large_cnn_text2sql_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("schema_aware_denoising_bart_large_cnn_text2sql_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|schema_aware_denoising_bart_large_cnn_text2sql_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.9 GB|

## References

https://huggingface.co/shahrukhx01/schema-aware-denoising-bart-large-cnn-text2sql

## Included Models

- DocumentAssembler
- BartTransformer