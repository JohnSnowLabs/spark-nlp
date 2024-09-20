---
layout: model
title: Persian albert_persian_farsi_base_v2_clf_digimag_pipeline pipeline AlbertForSequenceClassification from m3hrdadfi
author: John Snow Labs
name: albert_persian_farsi_base_v2_clf_digimag_pipeline
date: 2024-09-09
tags: [fa, open_source, pipeline, onnx]
task: Text Classification
language: fa
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albert_persian_farsi_base_v2_clf_digimag_pipeline` is a Persian model originally trained by m3hrdadfi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_persian_farsi_base_v2_clf_digimag_pipeline_fa_5.5.0_3.0_1725889141824.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_persian_farsi_base_v2_clf_digimag_pipeline_fa_5.5.0_3.0_1725889141824.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("albert_persian_farsi_base_v2_clf_digimag_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("albert_persian_farsi_base_v2_clf_digimag_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_persian_farsi_base_v2_clf_digimag_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|68.6 MB|

## References

https://huggingface.co/m3hrdadfi/albert-fa-base-v2-clf-digimag

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertForSequenceClassification