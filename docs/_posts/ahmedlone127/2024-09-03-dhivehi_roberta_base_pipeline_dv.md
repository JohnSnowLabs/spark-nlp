---
layout: model
title: Dhivehi, Divehi, Maldivian dhivehi_roberta_base_pipeline pipeline RoBertaEmbeddings from shahukareem
author: John Snow Labs
name: dhivehi_roberta_base_pipeline
date: 2024-09-03
tags: [dv, open_source, pipeline, onnx]
task: Embeddings
language: dv
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`dhivehi_roberta_base_pipeline` is a Dhivehi, Divehi, Maldivian model originally trained by shahukareem.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dhivehi_roberta_base_pipeline_dv_5.5.0_3.0_1725382134425.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dhivehi_roberta_base_pipeline_dv_5.5.0_3.0_1725382134425.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("dhivehi_roberta_base_pipeline", lang = "dv")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("dhivehi_roberta_base_pipeline", lang = "dv")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dhivehi_roberta_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|dv|
|Size:|464.9 MB|

## References

https://huggingface.co/shahukareem/dhivehi-roberta-base

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings