---
layout: model
title: English pipeline_image_classifier_vit_iiif_manuscript_ ViTForImageClassification from davanstrien
author: John Snow Labs
name: pipeline_image_classifier_vit_iiif_manuscript_
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

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_iiif_manuscript_` is a English model originally trained by davanstrien.


## Predicted Entities

`3rd upper flyleaf verso`, `Blank leaf recto`, `3rd lower flyleaf verso`, `2nd lower flyleaf verso`, `2nd upper flyleaf verso`, `flyleaf`, `1st upper flyleaf verso`, `1st lower flyleaf verso`, `fol`, `cover`, `Lower flyleaf verso`, `Blank leaf verso`, `Upper flyleaf verso`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_iiif_manuscript__en_4.2.1_3.0_1665538056313.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_iiif_manuscript__en_4.2.1_3.0_1665538056313.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_iiif_manuscript_', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_iiif_manuscript_", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_iiif_manuscript_|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|322.0 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification