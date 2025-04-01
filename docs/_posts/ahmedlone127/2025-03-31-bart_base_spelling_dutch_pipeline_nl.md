---
layout: model
title: Dutch, Flemish bart_base_spelling_dutch_pipeline pipeline BartTransformer from antalvdb
author: John Snow Labs
name: bart_base_spelling_dutch_pipeline
date: 2025-03-31
tags: [nl, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: nl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bart_base_spelling_dutch_pipeline` is a Dutch, Flemish model originally trained by antalvdb.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bart_base_spelling_dutch_pipeline_nl_5.5.1_3.0_1743436112427.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bart_base_spelling_dutch_pipeline_nl_5.5.1_3.0_1743436112427.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bart_base_spelling_dutch_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bart_base_spelling_dutch_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bart_base_spelling_dutch_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|811.7 MB|

## References

https://huggingface.co/antalvdb/bart-base-spelling-nl

## Included Models

- DocumentAssembler
- BartTransformer