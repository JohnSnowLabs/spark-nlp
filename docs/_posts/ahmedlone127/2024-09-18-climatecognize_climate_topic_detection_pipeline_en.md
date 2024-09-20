---
layout: model
title: English climatecognize_climate_topic_detection_pipeline pipeline RoBertaForSequenceClassification from gorgilazarev3
author: John Snow Labs
name: climatecognize_climate_topic_detection_pipeline
date: 2024-09-18
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`climatecognize_climate_topic_detection_pipeline` is a English model originally trained by gorgilazarev3.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/climatecognize_climate_topic_detection_pipeline_en_5.5.0_3.0_1726628061920.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/climatecognize_climate_topic_detection_pipeline_en_5.5.0_3.0_1726628061920.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("climatecognize_climate_topic_detection_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("climatecognize_climate_topic_detection_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|climatecognize_climate_topic_detection_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|309.7 MB|

## References

https://huggingface.co/gorgilazarev3/climatecognize-climate-topic-detection

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification