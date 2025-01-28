---
layout: model
title: Arabic bert_base_qarib60_1790k_pipeline pipeline BertEmbeddings from ahmedabdelali
author: John Snow Labs
name: bert_base_qarib60_1790k_pipeline
date: 2025-01-28
tags: [ar, open_source, pipeline, onnx]
task: Embeddings
language: ar
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_qarib60_1790k_pipeline` is a Arabic model originally trained by ahmedabdelali.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_qarib60_1790k_pipeline_ar_5.5.1_3.0_1738096311314.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_qarib60_1790k_pipeline_ar_5.5.1_3.0_1738096311314.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_base_qarib60_1790k_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_base_qarib60_1790k_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_qarib60_1790k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|505.0 MB|

## References

https://huggingface.co/ahmedabdelali/bert-base-qarib60_1790k

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings