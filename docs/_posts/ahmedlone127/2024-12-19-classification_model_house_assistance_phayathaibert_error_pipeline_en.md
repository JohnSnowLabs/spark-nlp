---
layout: model
title: English classification_model_house_assistance_phayathaibert_error_pipeline pipeline CamemBertForSequenceClassification from lunarlist
author: John Snow Labs
name: classification_model_house_assistance_phayathaibert_error_pipeline
date: 2024-12-19
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained CamemBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`classification_model_house_assistance_phayathaibert_error_pipeline` is a English model originally trained by lunarlist.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classification_model_house_assistance_phayathaibert_error_pipeline_en_5.5.1_3.0_1734572692760.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classification_model_house_assistance_phayathaibert_error_pipeline_en_5.5.1_3.0_1734572692760.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("classification_model_house_assistance_phayathaibert_error_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("classification_model_house_assistance_phayathaibert_error_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classification_model_house_assistance_phayathaibert_error_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.0 GB|

## References

https://huggingface.co/lunarlist/classification-model-house-assistance-phayathaibert-error

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForSequenceClassification