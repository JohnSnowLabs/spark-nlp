---
layout: model
title: English bert_base_squad_v1_1_portuguese_ibama_v0_420240915001955_pipeline pipeline BertForQuestionAnswering from alcalazans
author: John Snow Labs
name: bert_base_squad_v1_1_portuguese_ibama_v0_420240915001955_pipeline
date: 2024-09-23
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_squad_v1_1_portuguese_ibama_v0_420240915001955_pipeline` is a English model originally trained by alcalazans.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_squad_v1_1_portuguese_ibama_v0_420240915001955_pipeline_en_5.5.0_3.0_1727127770037.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_squad_v1_1_portuguese_ibama_v0_420240915001955_pipeline_en_5.5.0_3.0_1727127770037.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_base_squad_v1_1_portuguese_ibama_v0_420240915001955_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_base_squad_v1_1_portuguese_ibama_v0_420240915001955_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_squad_v1_1_portuguese_ibama_v0_420240915001955_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|405.9 MB|

## References

https://huggingface.co/alcalazans/bert-base-squad-v1.1-pt-IBAMA_v0.420240915001955

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering