---
layout: model
title: English llama_model_overfitted_reg_gpt_pipeline pipeline MPNetEmbeddings from soksay
author: John Snow Labs
name: llama_model_overfitted_reg_gpt_pipeline
date: 2024-09-18
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

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`llama_model_overfitted_reg_gpt_pipeline` is a English model originally trained by soksay.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/llama_model_overfitted_reg_gpt_pipeline_en_5.5.0_3.0_1726675059230.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/llama_model_overfitted_reg_gpt_pipeline_en_5.5.0_3.0_1726675059230.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("llama_model_overfitted_reg_gpt_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("llama_model_overfitted_reg_gpt_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|llama_model_overfitted_reg_gpt_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.2 MB|

## References

https://huggingface.co/soksay/llama_model_overfitted_REG_GPT

## Included Models

- DocumentAssembler
- MPNetEmbeddings