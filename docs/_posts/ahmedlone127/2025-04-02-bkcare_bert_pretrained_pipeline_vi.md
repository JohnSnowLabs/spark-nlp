---
layout: model
title: Vietnamese bkcare_bert_pretrained_pipeline pipeline BertEmbeddings from BookingCare
author: John Snow Labs
name: bkcare_bert_pretrained_pipeline
date: 2025-04-02
tags: [vi, open_source, pipeline, onnx]
task: Embeddings
language: vi
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bkcare_bert_pretrained_pipeline` is a Vietnamese model originally trained by BookingCare.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bkcare_bert_pretrained_pipeline_vi_5.5.1_3.0_1743594134868.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bkcare_bert_pretrained_pipeline_vi_5.5.1_3.0_1743594134868.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bkcare_bert_pretrained_pipeline", lang = "vi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bkcare_bert_pretrained_pipeline", lang = "vi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bkcare_bert_pretrained_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|vi|
|Size:|407.0 MB|

## References

https://huggingface.co/BookingCare/bkcare-bert-pretrained

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings