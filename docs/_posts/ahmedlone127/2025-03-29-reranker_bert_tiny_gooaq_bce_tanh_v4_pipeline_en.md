---
layout: model
title: English reranker_bert_tiny_gooaq_bce_tanh_v4_pipeline pipeline BertForSequenceClassification from cross-encoder-testing
author: John Snow Labs
name: reranker_bert_tiny_gooaq_bce_tanh_v4_pipeline
date: 2025-03-29
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`reranker_bert_tiny_gooaq_bce_tanh_v4_pipeline` is a English model originally trained by cross-encoder-testing.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/reranker_bert_tiny_gooaq_bce_tanh_v4_pipeline_en_5.5.1_3.0_1743234041101.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/reranker_bert_tiny_gooaq_bce_tanh_v4_pipeline_en_5.5.1_3.0_1743234041101.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("reranker_bert_tiny_gooaq_bce_tanh_v4_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("reranker_bert_tiny_gooaq_bce_tanh_v4_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|reranker_bert_tiny_gooaq_bce_tanh_v4_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|16.8 MB|

## References

https://huggingface.co/cross-encoder-testing/reranker-bert-tiny-gooaq-bce-tanh-v4

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification