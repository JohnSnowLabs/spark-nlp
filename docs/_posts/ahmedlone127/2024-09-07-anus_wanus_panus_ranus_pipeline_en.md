---
layout: model
title: English anus_wanus_panus_ranus_pipeline pipeline DistilBertForSequenceClassification from namebobb
author: John Snow Labs
name: anus_wanus_panus_ranus_pipeline
date: 2024-09-07
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`anus_wanus_panus_ranus_pipeline` is a English model originally trained by namebobb.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/anus_wanus_panus_ranus_pipeline_en_5.5.0_3.0_1725674952365.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/anus_wanus_panus_ranus_pipeline_en_5.5.0_3.0_1725674952365.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("anus_wanus_panus_ranus_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("anus_wanus_panus_ranus_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|anus_wanus_panus_ranus_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.6 MB|

## References

https://huggingface.co/namebobb/anus-wanus-panus-ranus

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification