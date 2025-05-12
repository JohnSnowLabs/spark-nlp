---
layout: model
title: Thai bert_intent_customer_support_thai_pipeline pipeline BertForSequenceClassification from Porameht
author: John Snow Labs
name: bert_intent_customer_support_thai_pipeline
date: 2025-04-07
tags: [th, open_source, pipeline, onnx]
task: Text Classification
language: th
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_intent_customer_support_thai_pipeline` is a Thai model originally trained by Porameht.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_intent_customer_support_thai_pipeline_th_5.5.1_3.0_1744056902245.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_intent_customer_support_thai_pipeline_th_5.5.1_3.0_1744056902245.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_intent_customer_support_thai_pipeline", lang = "th")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_intent_customer_support_thai_pipeline", lang = "th")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_intent_customer_support_thai_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|th|
|Size:|667.4 MB|

## References

https://huggingface.co/Porameht/bert-intent-customer-support-th

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification