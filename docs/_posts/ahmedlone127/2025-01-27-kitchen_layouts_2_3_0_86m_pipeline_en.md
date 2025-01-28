---
layout: model
title: English kitchen_layouts_2_3_0_86m_pipeline pipeline ViTForImageClassification from JuanMa360
author: John Snow Labs
name: kitchen_layouts_2_3_0_86m_pipeline
date: 2025-01-27
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kitchen_layouts_2_3_0_86m_pipeline` is a English model originally trained by JuanMa360.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kitchen_layouts_2_3_0_86m_pipeline_en_5.5.1_3.0_1738021181795.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kitchen_layouts_2_3_0_86m_pipeline_en_5.5.1_3.0_1738021181795.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kitchen_layouts_2_3_0_86m_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kitchen_layouts_2_3_0_86m_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kitchen_layouts_2_3_0_86m_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.3 MB|

## References

https://huggingface.co/JuanMa360/kitchen-layouts-2.3.0-86M

## Included Models

- ImageAssembler
- ViTForImageClassification