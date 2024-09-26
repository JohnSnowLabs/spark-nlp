---
layout: model
title: English albert_base_chinese_finetuned_qqp_fhtm_5x_weak_pipeline pipeline BertForSequenceClassification from r10521708
author: John Snow Labs
name: albert_base_chinese_finetuned_qqp_fhtm_5x_weak_pipeline
date: 2024-09-25
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albert_base_chinese_finetuned_qqp_fhtm_5x_weak_pipeline` is a English model originally trained by r10521708.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_base_chinese_finetuned_qqp_fhtm_5x_weak_pipeline_en_5.5.0_3.0_1727301439790.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_base_chinese_finetuned_qqp_fhtm_5x_weak_pipeline_en_5.5.0_3.0_1727301439790.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("albert_base_chinese_finetuned_qqp_fhtm_5x_weak_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("albert_base_chinese_finetuned_qqp_fhtm_5x_weak_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_base_chinese_finetuned_qqp_fhtm_5x_weak_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|39.8 MB|

## References

https://huggingface.co/r10521708/albert-base-chinese-finetuned-qqp-FHTM-5x-weak

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification