---
layout: model
title: Italian distilgpt2_psy_ita_pipeline pipeline GPT2Transformer from misterkilgore
author: John Snow Labs
name: distilgpt2_psy_ita_pipeline
date: 2025-04-07
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

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilgpt2_psy_ita_pipeline` is a Italian model originally trained by misterkilgore.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilgpt2_psy_ita_pipeline_it_5.5.1_3.0_1743997171020.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilgpt2_psy_ita_pipeline_it_5.5.1_3.0_1743997171020.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilgpt2_psy_ita_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilgpt2_psy_ita_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilgpt2_psy_ita_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|308.4 MB|

## References

https://huggingface.co/misterkilgore/distilgpt2-psy-ita

## Included Models

- DocumentAssembler
- GPT2Transformer