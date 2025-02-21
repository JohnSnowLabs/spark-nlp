---
layout: model
title: Czech czegpt_2_pipeline pipeline GPT2Transformer from MU-NLPC
author: John Snow Labs
name: czegpt_2_pipeline
date: 2025-01-24
tags: [cs, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: cs
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`czegpt_2_pipeline` is a Czech model originally trained by MU-NLPC.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/czegpt_2_pipeline_cs_5.5.1_3.0_1737732005591.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/czegpt_2_pipeline_cs_5.5.1_3.0_1737732005591.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("czegpt_2_pipeline", lang = "cs")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("czegpt_2_pipeline", lang = "cs")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|czegpt_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|cs|
|Size:|299.1 MB|

## References

https://huggingface.co/MU-NLPC/CzeGPT-2

## Included Models

- DocumentAssembler
- GPT2Transformer