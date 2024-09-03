---
layout: model
title: English feel_italian_finetuned_pro_emit_big6_pipeline pipeline CamemBertForSequenceClassification from lupobricco
author: John Snow Labs
name: feel_italian_finetuned_pro_emit_big6_pipeline
date: 2024-09-03
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`feel_italian_finetuned_pro_emit_big6_pipeline` is a English model originally trained by lupobricco.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/feel_italian_finetuned_pro_emit_big6_pipeline_en_5.5.0_3.0_1725324976260.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/feel_italian_finetuned_pro_emit_big6_pipeline_en_5.5.0_3.0_1725324976260.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("feel_italian_finetuned_pro_emit_big6_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("feel_italian_finetuned_pro_emit_big6_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|feel_italian_finetuned_pro_emit_big6_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|399.1 MB|

## References

https://huggingface.co/lupobricco/feel_it_finetuned_pro_emit_big6

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForSequenceClassification