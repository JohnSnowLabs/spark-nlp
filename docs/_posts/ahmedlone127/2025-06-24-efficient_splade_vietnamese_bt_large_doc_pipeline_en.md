---
layout: model
title: English efficient_splade_vietnamese_bt_large_doc_pipeline pipeline DistilBertEmbeddings from naver
author: John Snow Labs
name: efficient_splade_vietnamese_bt_large_doc_pipeline
date: 2025-06-24
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

Pretrained DistilBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`efficient_splade_vietnamese_bt_large_doc_pipeline` is a English model originally trained by naver.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/efficient_splade_vietnamese_bt_large_doc_pipeline_en_5.5.1_3.0_1750780301034.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/efficient_splade_vietnamese_bt_large_doc_pipeline_en_5.5.1_3.0_1750780301034.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("efficient_splade_vietnamese_bt_large_doc_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("efficient_splade_vietnamese_bt_large_doc_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|efficient_splade_vietnamese_bt_large_doc_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.4 MB|

## References

https://huggingface.co/naver/efficient-splade-VI-BT-large-doc

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertEmbeddings