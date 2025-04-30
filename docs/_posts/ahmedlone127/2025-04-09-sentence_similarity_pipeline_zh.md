---
layout: model
title: Chinese sentence_similarity_pipeline pipeline BertForSequenceClassification from roberthsu2003
author: John Snow Labs
name: sentence_similarity_pipeline
date: 2025-04-09
tags: [zh, open_source, pipeline, onnx]
task: Text Classification
language: zh
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sentence_similarity_pipeline` is a Chinese model originally trained by roberthsu2003.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_similarity_pipeline_zh_5.5.1_3.0_1744188306404.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentence_similarity_pipeline_zh_5.5.1_3.0_1744188306404.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sentence_similarity_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sentence_similarity_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentence_similarity_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|383.3 MB|

## References

https://huggingface.co/roberthsu2003/sentence_similarity

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification