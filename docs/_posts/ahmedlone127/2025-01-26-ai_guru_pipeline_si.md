---
layout: model
title: Sinhala, Sinhalese ai_guru_pipeline pipeline GPT2Transformer from enzer1992
author: John Snow Labs
name: ai_guru_pipeline
date: 2025-01-26
tags: [si, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: si
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ai_guru_pipeline` is a Sinhala, Sinhalese model originally trained by enzer1992.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ai_guru_pipeline_si_5.5.1_3.0_1737916173820.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ai_guru_pipeline_si_5.5.1_3.0_1737916173820.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ai_guru_pipeline", lang = "si")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ai_guru_pipeline", lang = "si")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ai_guru_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|si|
|Size:|467.8 MB|

## References

https://huggingface.co/enzer1992/AI-Guru

## Included Models

- DocumentAssembler
- GPT2Transformer