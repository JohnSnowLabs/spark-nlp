---
layout: model
title: Oriya (macrolanguage) odia_topic_all_doc_pipeline pipeline BertForSequenceClassification from l3cube-pune
author: John Snow Labs
name: odia_topic_all_doc_pipeline
date: 2024-09-26
tags: [or, open_source, pipeline, onnx]
task: Text Classification
language: or
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`odia_topic_all_doc_pipeline` is a Oriya (macrolanguage) model originally trained by l3cube-pune.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/odia_topic_all_doc_pipeline_or_5.5.0_3.0_1727369060155.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/odia_topic_all_doc_pipeline_or_5.5.0_3.0_1727369060155.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("odia_topic_all_doc_pipeline", lang = "or")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("odia_topic_all_doc_pipeline", lang = "or")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|odia_topic_all_doc_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|or|
|Size:|892.7 MB|

## References

https://huggingface.co/l3cube-pune/odia-topic-all-doc

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification