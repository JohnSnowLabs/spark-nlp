---
layout: model
title: English pipeline_image_classifier_vit_violation_classification_bantai__v80ep ViTForImageClassification from AykeeSalazar
author: John Snow Labs
name: pipeline_image_classifier_vit_violation_classification_bantai__v80ep
date: 2022-10-12
tags: [vit, en, images, open_source, pipeline]
task: Image Classification
language: en
edition: Spark NLP 4.2.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_violation_classification_bantai__v80ep` is a English model originally trained by AykeeSalazar.


## Predicted Entities

`Public Smoking`, `Public-Drinking`, `ambiguous`, `non-violation`


## Predicted Entities

`Public Smoking`, `Public-Drinking`, `ambiguous`, `non-violation`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_violation_classification_bantai__v80ep_en_4.2.1_3.0_1665536850423.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_violation_classification_bantai__v80ep_en_4.2.1_3.0_1665536850423.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_violation_classification_bantai__v80ep', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_violation_classification_bantai__v80ep", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_violation_classification_bantai__v80ep|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.9 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification