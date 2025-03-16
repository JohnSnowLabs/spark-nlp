---
layout: model
title: English topk_10epoch_nepal_bhasa_60_pruned_pipeline pipeline BertForQuestionAnswering from xihajun
author: John Snow Labs
name: topk_10epoch_nepal_bhasa_60_pruned_pipeline
date: 2025-01-24
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`topk_10epoch_nepal_bhasa_60_pruned_pipeline` is a English model originally trained by xihajun.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/topk_10epoch_nepal_bhasa_60_pruned_pipeline_en_5.5.1_3.0_1737752172061.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/topk_10epoch_nepal_bhasa_60_pruned_pipeline_en_5.5.1_3.0_1737752172061.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("topk_10epoch_nepal_bhasa_60_pruned_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("topk_10epoch_nepal_bhasa_60_pruned_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|topk_10epoch_nepal_bhasa_60_pruned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/xihajun/topK_10epoch_new_60_pruned

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering