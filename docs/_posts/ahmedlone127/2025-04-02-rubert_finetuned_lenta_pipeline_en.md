---
layout: model
title: English rubert_finetuned_lenta_pipeline pipeline BertEmbeddings from matvej-melikhov
author: John Snow Labs
name: rubert_finetuned_lenta_pipeline
date: 2025-04-02
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`rubert_finetuned_lenta_pipeline` is a English model originally trained by matvej-melikhov.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/rubert_finetuned_lenta_pipeline_en_5.5.1_3.0_1743553671662.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/rubert_finetuned_lenta_pipeline_en_5.5.1_3.0_1743553671662.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("rubert_finetuned_lenta_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("rubert_finetuned_lenta_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rubert_finetuned_lenta_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|667.1 MB|

## References

https://huggingface.co/matvej-melikhov/ruBERT-finetuned-lenta

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings