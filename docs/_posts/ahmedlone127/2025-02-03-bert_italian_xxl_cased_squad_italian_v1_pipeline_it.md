---
layout: model
title: Italian bert_italian_xxl_cased_squad_italian_v1_pipeline pipeline BertForQuestionAnswering from fabgraziano
author: John Snow Labs
name: bert_italian_xxl_cased_squad_italian_v1_pipeline
date: 2025-02-03
tags: [it, open_source, pipeline, onnx]
task: Question Answering
language: it
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_italian_xxl_cased_squad_italian_v1_pipeline` is a Italian model originally trained by fabgraziano.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_italian_xxl_cased_squad_italian_v1_pipeline_it_5.5.1_3.0_1738559206179.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_italian_xxl_cased_squad_italian_v1_pipeline_it_5.5.1_3.0_1738559206179.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_italian_xxl_cased_squad_italian_v1_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_italian_xxl_cased_squad_italian_v1_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_italian_xxl_cased_squad_italian_v1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|412.6 MB|

## References

https://huggingface.co/fabgraziano/bert-italian-xxl-cased_squad-it_v1

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering