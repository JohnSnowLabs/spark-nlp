---
layout: model
title: English classifier_chapter4_caozhejiong_pipeline pipeline DistilBertForSequenceClassification from caozhejiong
author: John Snow Labs
name: classifier_chapter4_caozhejiong_pipeline
date: 2025-04-01
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`classifier_chapter4_caozhejiong_pipeline` is a English model originally trained by caozhejiong.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifier_chapter4_caozhejiong_pipeline_en_5.5.1_3.0_1743538881037.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classifier_chapter4_caozhejiong_pipeline_en_5.5.1_3.0_1743538881037.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("classifier_chapter4_caozhejiong_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("classifier_chapter4_caozhejiong_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifier_chapter4_caozhejiong_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/caozhejiong/classifier-chapter4

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification