---
layout: model
title: English image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_q2_jlbar TFSwinForImageClassification from q2-jlbar
author: John Snow Labs
name: pipeline_image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_q2_jlbar
date: 2023-03-23
tags: [swin, en, image, open_source, pipeline, image_classification, imagenet]
task: Image Classification
language: en
edition: Spark NLP 4.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained  Swin  pipeline, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_q2_jlbar` is a English model originally trained by q2-jlbar.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_q2_jlbar_en_4.4.0_3.0_1679592933286.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_q2_jlbar_en_4.4.0_3.0_1679592933286.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_q2_jlbar', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_q2_jlbar", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_swin_swin_tiny_patch4_window7_224_finetuned_eurosat_by_q2_jlbar|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|210.5 MB|

## Included Models

- ImageAssembler
- SwinForImageClassification