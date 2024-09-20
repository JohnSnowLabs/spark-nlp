---
layout: model
title: English autotrain_nrfmc_2jyt8_pipeline pipeline MPNetEmbeddings from bhawanahettiarachchi
author: John Snow Labs
name: autotrain_nrfmc_2jyt8_pipeline
date: 2024-09-02
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`autotrain_nrfmc_2jyt8_pipeline` is a English model originally trained by bhawanahettiarachchi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/autotrain_nrfmc_2jyt8_pipeline_en_5.5.0_3.0_1725258290753.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/autotrain_nrfmc_2jyt8_pipeline_en_5.5.0_3.0_1725258290753.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("autotrain_nrfmc_2jyt8_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("autotrain_nrfmc_2jyt8_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|autotrain_nrfmc_2jyt8_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.7 MB|

## References

https://huggingface.co/bhawanahettiarachchi/autotrain-nrfmc-2jyt8

## Included Models

- DocumentAssembler
- MPNetEmbeddings