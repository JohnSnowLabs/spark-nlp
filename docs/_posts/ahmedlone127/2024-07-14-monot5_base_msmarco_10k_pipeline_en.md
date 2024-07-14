---
layout: model
title: English monot5_base_msmarco_10k_pipeline pipeline T5Transformer from castorini
author: John Snow Labs
name: monot5_base_msmarco_10k_pipeline
date: 2024-07-14
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`monot5_base_msmarco_10k_pipeline` is a English model originally trained by castorini.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/monot5_base_msmarco_10k_pipeline_en_5.4.1_3.0_1720968849208.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/monot5_base_msmarco_10k_pipeline_en_5.4.1_3.0_1720968849208.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("monot5_base_msmarco_10k_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("monot5_base_msmarco_10k_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|monot5_base_msmarco_10k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|520.8 MB|

## References

https://huggingface.co/castorini/monot5-base-msmarco-10k

## Included Models

- DocumentAssembler
- T5Transformer