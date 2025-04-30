---
layout: model
title: Italian legit_bart_lsg_4096_pipeline pipeline BartTransformer from morenolq
author: John Snow Labs
name: legit_bart_lsg_4096_pipeline
date: 2025-04-04
tags: [it, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
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

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`legit_bart_lsg_4096_pipeline` is a Italian model originally trained by morenolq.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/legit_bart_lsg_4096_pipeline_it_5.5.1_3.0_1743754628403.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/legit_bart_lsg_4096_pipeline_it_5.5.1_3.0_1743754628403.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("legit_bart_lsg_4096_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("legit_bart_lsg_4096_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legit_bart_lsg_4096_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|843.5 MB|

## References

https://huggingface.co/morenolq/LEGIT-BART-LSG-4096

## Included Models

- DocumentAssembler
- BartTransformer