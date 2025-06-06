---
layout: model
title: Italian legit_scratch_bart_pipeline pipeline BartTransformer from morenolq
author: John Snow Labs
name: legit_scratch_bart_pipeline
date: 2025-04-08
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

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`legit_scratch_bart_pipeline` is a Italian model originally trained by morenolq.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/legit_scratch_bart_pipeline_it_5.5.1_3.0_1744079097801.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/legit_scratch_bart_pipeline_it_5.5.1_3.0_1744079097801.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("legit_scratch_bart_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("legit_scratch_bart_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legit_scratch_bart_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|827.7 MB|

## References

https://huggingface.co/morenolq/LEGIT-SCRATCH-BART

## Included Models

- DocumentAssembler
- BartTransformer