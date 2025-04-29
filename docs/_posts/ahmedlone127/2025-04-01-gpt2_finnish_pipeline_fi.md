---
layout: model
title: Finnish gpt2_finnish_pipeline pipeline GPT2Transformer from Finnish-NLP
author: John Snow Labs
name: gpt2_finnish_pipeline
date: 2025-04-01
tags: [fi, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: fi
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpt2_finnish_pipeline` is a Finnish model originally trained by Finnish-NLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpt2_finnish_pipeline_fi_5.5.1_3.0_1743535287656.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpt2_finnish_pipeline_fi_5.5.1_3.0_1743535287656.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gpt2_finnish_pipeline", lang = "fi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gpt2_finnish_pipeline", lang = "fi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpt2_finnish_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fi|
|Size:|466.7 MB|

## References

https://huggingface.co/Finnish-NLP/gpt2-finnish

## Included Models

- DocumentAssembler
- GPT2Transformer