---
layout: model
title: Thai mt5_budget_seq2seq_2_pipeline pipeline T5Transformer from napatswift
author: John Snow Labs
name: mt5_budget_seq2seq_2_pipeline
date: 2024-08-21
tags: [th, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: th
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_budget_seq2seq_2_pipeline` is a Thai model originally trained by napatswift.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_budget_seq2seq_2_pipeline_th_5.4.2_3.0_1724259698483.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_budget_seq2seq_2_pipeline_th_5.4.2_3.0_1724259698483.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_budget_seq2seq_2_pipeline", lang = "th")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_budget_seq2seq_2_pipeline", lang = "th")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_budget_seq2seq_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|th|
|Size:|969.5 MB|

## References

https://huggingface.co/napatswift/mt5-budget-seq2seq-2

## Included Models

- DocumentAssembler
- T5Transformer