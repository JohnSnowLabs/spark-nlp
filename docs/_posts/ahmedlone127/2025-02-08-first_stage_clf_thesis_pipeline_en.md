---
layout: model
title: English first_stage_clf_thesis_pipeline pipeline RoBertaForSequenceClassification from janbakker
author: John Snow Labs
name: first_stage_clf_thesis_pipeline
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`first_stage_clf_thesis_pipeline` is a English model originally trained by janbakker.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/first_stage_clf_thesis_pipeline_en_5.5.1_3.0_1738987056512.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/first_stage_clf_thesis_pipeline_en_5.5.1_3.0_1738987056512.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("first_stage_clf_thesis_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("first_stage_clf_thesis_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|first_stage_clf_thesis_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|432.4 MB|

## References

https://huggingface.co/janbakker/first-stage-clf-thesis

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification