---
layout: model
title: English vision_transformers_spain_oriya_italy_fan_pipeline pipeline ViTForImageClassification from jeffboudier
author: John Snow Labs
name: vision_transformers_spain_oriya_italy_fan_pipeline
date: 2025-01-31
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`vision_transformers_spain_oriya_italy_fan_pipeline` is a English model originally trained by jeffboudier.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/vision_transformers_spain_oriya_italy_fan_pipeline_en_5.5.1_3.0_1738307939747.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/vision_transformers_spain_oriya_italy_fan_pipeline_en_5.5.1_3.0_1738307939747.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("vision_transformers_spain_oriya_italy_fan_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("vision_transformers_spain_oriya_italy_fan_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|vision_transformers_spain_oriya_italy_fan_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.3 MB|

## References

https://huggingface.co/jeffboudier/vision-transformers-spain-or-italy-fan

## Included Models

- ImageAssembler
- ViTForImageClassification