---
layout: model
title: English progan_ai_vs_real_image_detection_pipeline pipeline ViTForImageClassification from ongtrandong2
author: John Snow Labs
name: progan_ai_vs_real_image_detection_pipeline
date: 2025-02-06
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`progan_ai_vs_real_image_detection_pipeline` is a English model originally trained by ongtrandong2.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/progan_ai_vs_real_image_detection_pipeline_en_5.5.1_3.0_1738880124624.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/progan_ai_vs_real_image_detection_pipeline_en_5.5.1_3.0_1738880124624.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("progan_ai_vs_real_image_detection_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("progan_ai_vs_real_image_detection_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|progan_ai_vs_real_image_detection_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.3 MB|

## References

https://huggingface.co/ongtrandong2/progan_ai_vs_real_image_detection

## Included Models

- ImageAssembler
- ViTForImageClassification