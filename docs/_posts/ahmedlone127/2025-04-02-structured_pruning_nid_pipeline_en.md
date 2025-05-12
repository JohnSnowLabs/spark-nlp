---
layout: model
title: English structured_pruning_nid_pipeline pipeline DistilBertForSequenceClassification from gates04
author: John Snow Labs
name: structured_pruning_nid_pipeline
date: 2025-04-02
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`structured_pruning_nid_pipeline` is a English model originally trained by gates04.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/structured_pruning_nid_pipeline_en_5.5.1_3.0_1743562361399.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/structured_pruning_nid_pipeline_en_5.5.1_3.0_1743562361399.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("structured_pruning_nid_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("structured_pruning_nid_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|structured_pruning_nid_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|246.1 MB|

## References

https://huggingface.co/gates04/Structured-Pruning-NID

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification