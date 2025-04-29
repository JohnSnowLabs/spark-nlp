---
layout: model
title: English distilgpt2_finetuned_wikitext2_mbrogan55_pipeline pipeline GPT2Transformer from Mbrogan55
author: John Snow Labs
name: distilgpt2_finetuned_wikitext2_mbrogan55_pipeline
date: 2025-02-02
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilgpt2_finetuned_wikitext2_mbrogan55_pipeline` is a English model originally trained by Mbrogan55.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilgpt2_finetuned_wikitext2_mbrogan55_pipeline_en_5.5.1_3.0_1738527470752.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilgpt2_finetuned_wikitext2_mbrogan55_pipeline_en_5.5.1_3.0_1738527470752.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilgpt2_finetuned_wikitext2_mbrogan55_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilgpt2_finetuned_wikitext2_mbrogan55_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilgpt2_finetuned_wikitext2_mbrogan55_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|308.4 MB|

## References

https://huggingface.co/Mbrogan55/distilgpt2-finetuned-wikitext2

## Included Models

- DocumentAssembler
- GPT2Transformer