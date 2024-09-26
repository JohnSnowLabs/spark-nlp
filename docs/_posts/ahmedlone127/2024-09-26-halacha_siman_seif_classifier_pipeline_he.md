---
layout: model
title: Hebrew halacha_siman_seif_classifier_pipeline pipeline BertForSequenceClassification from sivan22
author: John Snow Labs
name: halacha_siman_seif_classifier_pipeline
date: 2024-09-26
tags: [he, open_source, pipeline, onnx]
task: Text Classification
language: he
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`halacha_siman_seif_classifier_pipeline` is a Hebrew model originally trained by sivan22.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/halacha_siman_seif_classifier_pipeline_he_5.5.0_3.0_1727331507902.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/halacha_siman_seif_classifier_pipeline_he_5.5.0_3.0_1727331507902.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("halacha_siman_seif_classifier_pipeline", lang = "he")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("halacha_siman_seif_classifier_pipeline", lang = "he")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|halacha_siman_seif_classifier_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|he|
|Size:|704.2 MB|

## References

https://huggingface.co/sivan22/halacha-siman-seif-classifier

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification