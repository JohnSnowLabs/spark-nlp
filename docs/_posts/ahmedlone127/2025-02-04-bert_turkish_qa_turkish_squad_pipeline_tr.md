---
layout: model
title: Turkish bert_turkish_qa_turkish_squad_pipeline pipeline BertForQuestionAnswering from anilguven
author: John Snow Labs
name: bert_turkish_qa_turkish_squad_pipeline
date: 2025-02-04
tags: [tr, open_source, pipeline, onnx]
task: Question Answering
language: tr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_turkish_qa_turkish_squad_pipeline` is a Turkish model originally trained by anilguven.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_turkish_qa_turkish_squad_pipeline_tr_5.5.1_3.0_1738669180773.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_turkish_qa_turkish_squad_pipeline_tr_5.5.1_3.0_1738669180773.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_turkish_qa_turkish_squad_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_turkish_qa_turkish_squad_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_turkish_qa_turkish_squad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|412.5 MB|

## References

https://huggingface.co/anilguven/bert_tr_qa_turkish_squad

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering