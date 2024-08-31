---
layout: model
title: English sent_efficient_splade_vietnamese_bt_large_query_naver_pipeline pipeline BertSentenceEmbeddings from naver
author: John Snow Labs
name: sent_efficient_splade_vietnamese_bt_large_query_naver_pipeline
date: 2024-08-31
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_efficient_splade_vietnamese_bt_large_query_naver_pipeline` is a English model originally trained by naver.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_efficient_splade_vietnamese_bt_large_query_naver_pipeline_en_5.4.2_3.0_1725121715343.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_efficient_splade_vietnamese_bt_large_query_naver_pipeline_en_5.4.2_3.0_1725121715343.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_efficient_splade_vietnamese_bt_large_query_naver_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_efficient_splade_vietnamese_bt_large_query_naver_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_efficient_splade_vietnamese_bt_large_query_naver_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|17.2 MB|

## References

https://huggingface.co/naver/efficient-splade-VI-BT-large-query

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings