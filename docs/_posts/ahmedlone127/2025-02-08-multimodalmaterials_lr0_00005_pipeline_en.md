---
layout: model
title: English multimodalmaterials_lr0_00005_pipeline pipeline RoBertaForSequenceClassification from rose-e-wang
author: John Snow Labs
name: multimodalmaterials_lr0_00005_pipeline
date: 2025-02-08
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multimodalmaterials_lr0_00005_pipeline` is a English model originally trained by rose-e-wang.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multimodalmaterials_lr0_00005_pipeline_en_5.5.1_3.0_1738991716698.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multimodalmaterials_lr0_00005_pipeline_en_5.5.1_3.0_1738991716698.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("multimodalmaterials_lr0_00005_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("multimodalmaterials_lr0_00005_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multimodalmaterials_lr0_00005_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/rose-e-wang/multimodalMaterials_lr0.00005

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification