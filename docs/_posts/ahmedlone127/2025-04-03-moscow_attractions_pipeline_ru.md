---
layout: model
title: Russian moscow_attractions_pipeline pipeline ViTForImageClassification from Poliandr
author: John Snow Labs
name: moscow_attractions_pipeline
date: 2025-04-03
tags: [ru, open_source, pipeline, onnx]
task: Image Classification
language: ru
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`moscow_attractions_pipeline` is a Russian model originally trained by Poliandr.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/moscow_attractions_pipeline_ru_5.5.1_3.0_1743678658496.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/moscow_attractions_pipeline_ru_5.5.1_3.0_1743678658496.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("moscow_attractions_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("moscow_attractions_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|moscow_attractions_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|321.3 MB|

## References

https://huggingface.co/Poliandr/moscow-attractions

## Included Models

- ImageAssembler
- ViTForImageClassification