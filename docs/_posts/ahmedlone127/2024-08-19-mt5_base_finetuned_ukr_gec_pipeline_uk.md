---
layout: model
title: Ukrainian mt5_base_finetuned_ukr_gec_pipeline pipeline T5Transformer from schhwmn
author: John Snow Labs
name: mt5_base_finetuned_ukr_gec_pipeline
date: 2024-08-19
tags: [uk, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: uk
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_base_finetuned_ukr_gec_pipeline` is a Ukrainian model originally trained by schhwmn.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_base_finetuned_ukr_gec_pipeline_uk_5.4.2_3.0_1724034238685.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_base_finetuned_ukr_gec_pipeline_uk_5.4.2_3.0_1724034238685.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_base_finetuned_ukr_gec_pipeline", lang = "uk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_base_finetuned_ukr_gec_pipeline", lang = "uk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_base_finetuned_ukr_gec_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|uk|
|Size:|2.3 GB|

## References

https://huggingface.co/schhwmn/mt5-base-finetuned-ukr-gec

## Included Models

- DocumentAssembler
- T5Transformer