---
layout: model
title: Portuguese bert_squad_portuguese_pipeline pipeline BertForQuestionAnswering from rhaymison
author: John Snow Labs
name: bert_squad_portuguese_pipeline
date: 2024-09-02
tags: [pt, open_source, pipeline, onnx]
task: Question Answering
language: pt
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_squad_portuguese_pipeline` is a Portuguese model originally trained by rhaymison.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_squad_portuguese_pipeline_pt_5.5.0_3.0_1725313145411.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_squad_portuguese_pipeline_pt_5.5.0_3.0_1725313145411.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_squad_portuguese_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_squad_portuguese_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_squad_portuguese_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|405.9 MB|

## References

https://huggingface.co/rhaymison/bert-squad-portuguese

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering