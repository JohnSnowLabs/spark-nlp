---
layout: model
title: English all_minilm_l6_v2_finetuned_squad_pipeline pipeline BertForQuestionAnswering from Sybghat
author: John Snow Labs
name: all_minilm_l6_v2_finetuned_squad_pipeline
date: 2024-09-20
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`all_minilm_l6_v2_finetuned_squad_pipeline` is a English model originally trained by Sybghat.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/all_minilm_l6_v2_finetuned_squad_pipeline_en_5.5.0_3.0_1726834335735.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/all_minilm_l6_v2_finetuned_squad_pipeline_en_5.5.0_3.0_1726834335735.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("all_minilm_l6_v2_finetuned_squad_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("all_minilm_l6_v2_finetuned_squad_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|all_minilm_l6_v2_finetuned_squad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|84.2 MB|

## References

https://huggingface.co/Sybghat/all-MiniLM-L6-v2-finetuned-squad

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering