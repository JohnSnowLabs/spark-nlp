---
layout: model
title: Portuguese gpt2_medium_squadv11_portuguese_pipeline pipeline GPT2Transformer from egonrp
author: John Snow Labs
name: gpt2_medium_squadv11_portuguese_pipeline
date: 2025-01-31
tags: [pt, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: pt
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpt2_medium_squadv11_portuguese_pipeline` is a Portuguese model originally trained by egonrp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpt2_medium_squadv11_portuguese_pipeline_pt_5.5.1_3.0_1738357114686.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpt2_medium_squadv11_portuguese_pipeline_pt_5.5.1_3.0_1738357114686.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gpt2_medium_squadv11_portuguese_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gpt2_medium_squadv11_portuguese_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpt2_medium_squadv11_portuguese_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|1.3 GB|

## References

https://huggingface.co/egonrp/gpt2-medium-squadv11-portuguese

## Included Models

- DocumentAssembler
- GPT2Transformer