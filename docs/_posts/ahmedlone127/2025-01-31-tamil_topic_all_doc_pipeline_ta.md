---
layout: model
title: Tamil tamil_topic_all_doc_pipeline pipeline BertForSequenceClassification from l3cube-pune
author: John Snow Labs
name: tamil_topic_all_doc_pipeline
date: 2025-01-31
tags: [ta, open_source, pipeline, onnx]
task: Text Classification
language: ta
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tamil_topic_all_doc_pipeline` is a Tamil model originally trained by l3cube-pune.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tamil_topic_all_doc_pipeline_ta_5.5.1_3.0_1738326506239.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tamil_topic_all_doc_pipeline_ta_5.5.1_3.0_1738326506239.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tamil_topic_all_doc_pipeline", lang = "ta")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tamil_topic_all_doc_pipeline", lang = "ta")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tamil_topic_all_doc_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ta|
|Size:|892.9 MB|

## References

https://huggingface.co/l3cube-pune/tamil-topic-all-doc

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification