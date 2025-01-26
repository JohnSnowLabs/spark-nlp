---
layout: model
title: English skin_cancer_image_classification_pipeline pipeline ViTForImageClassification from Anwarkh1
author: John Snow Labs
name: skin_cancer_image_classification_pipeline
date: 2025-01-26
tags: [en, open_source, pipeline, onnx]
task: Image Classification
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`skin_cancer_image_classification_pipeline` is a English model originally trained by Anwarkh1.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/skin_cancer_image_classification_pipeline_en_5.5.1_3.0_1737926404768.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/skin_cancer_image_classification_pipeline_en_5.5.1_3.0_1737926404768.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("skin_cancer_image_classification_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("skin_cancer_image_classification_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|skin_cancer_image_classification_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.3 MB|

## References

https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification

## Included Models

- ImageAssembler
- ViTForImageClassification