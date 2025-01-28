---
layout: model
title: English slm_custom_model_5_rk_pipeline pipeline T5Transformer from karthikeyan-r
author: John Snow Labs
name: slm_custom_model_5_rk_pipeline
date: 2025-01-28
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`slm_custom_model_5_rk_pipeline` is a English model originally trained by karthikeyan-r.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/slm_custom_model_5_rk_pipeline_en_5.5.1_3.0_1738091394845.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/slm_custom_model_5_rk_pipeline_en_5.5.1_3.0_1738091394845.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("slm_custom_model_5_rk_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("slm_custom_model_5_rk_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|slm_custom_model_5_rk_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|277.5 MB|

## References

https://huggingface.co/karthikeyan-r/slm-custom-model_5_RK

## Included Models

- DocumentAssembler
- T5Transformer