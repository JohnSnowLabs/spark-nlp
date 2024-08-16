---
layout: model
title: English mt5_small_finetuned_amazon_english_spanish_cjd536_pipeline pipeline T5Transformer from cjd536
author: John Snow Labs
name: mt5_small_finetuned_amazon_english_spanish_cjd536_pipeline
date: 2024-08-16
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_small_finetuned_amazon_english_spanish_cjd536_pipeline` is a English model originally trained by cjd536.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_small_finetuned_amazon_english_spanish_cjd536_pipeline_en_5.4.2_3.0_1723820714256.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_small_finetuned_amazon_english_spanish_cjd536_pipeline_en_5.4.2_3.0_1723820714256.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_small_finetuned_amazon_english_spanish_cjd536_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_small_finetuned_amazon_english_spanish_cjd536_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_small_finetuned_amazon_english_spanish_cjd536_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.2 GB|

## References

https://huggingface.co/cjd536/mt5-small-finetuned-amazon-en-es

## Included Models

- DocumentAssembler
- T5Transformer