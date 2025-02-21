---
layout: model
title: Turkish gpt2_turkish_900m_pipeline pipeline GPT2Transformer from cenkersisman
author: John Snow Labs
name: gpt2_turkish_900m_pipeline
date: 2025-01-26
tags: [tr, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: tr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpt2_turkish_900m_pipeline` is a Turkish model originally trained by cenkersisman.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpt2_turkish_900m_pipeline_tr_5.5.1_3.0_1737913373026.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpt2_turkish_900m_pipeline_tr_5.5.1_3.0_1737913373026.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gpt2_turkish_900m_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gpt2_turkish_900m_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpt2_turkish_900m_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|467.1 MB|

## References

https://huggingface.co/cenkersisman/gpt2-turkish-900m

## Included Models

- DocumentAssembler
- GPT2Transformer