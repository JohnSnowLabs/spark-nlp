---
layout: model
title: English esnli_limited_e_10_alpha_0_5_pipeline pipeline T5Transformer from harish
author: John Snow Labs
name: esnli_limited_e_10_alpha_0_5_pipeline
date: 2024-08-24
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`esnli_limited_e_10_alpha_0_5_pipeline` is a English model originally trained by harish.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/esnli_limited_e_10_alpha_0_5_pipeline_en_5.4.2_3.0_1724504847195.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/esnli_limited_e_10_alpha_0_5_pipeline_en_5.4.2_3.0_1724504847195.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("esnli_limited_e_10_alpha_0_5_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("esnli_limited_e_10_alpha_0_5_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|esnli_limited_e_10_alpha_0_5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|986.7 MB|

## References

https://huggingface.co/harish/eSNLI-limited-e-10-alpha-0-5

## Included Models

- DocumentAssembler
- T5Transformer