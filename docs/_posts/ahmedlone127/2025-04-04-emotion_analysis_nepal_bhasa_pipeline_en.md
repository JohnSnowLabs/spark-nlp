---
layout: model
title: English emotion_analysis_nepal_bhasa_pipeline pipeline RoBertaForSequenceClassification from Pavan97
author: John Snow Labs
name: emotion_analysis_nepal_bhasa_pipeline
date: 2025-04-04
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`emotion_analysis_nepal_bhasa_pipeline` is a English model originally trained by Pavan97.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/emotion_analysis_nepal_bhasa_pipeline_en_5.5.1_3.0_1743728675125.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/emotion_analysis_nepal_bhasa_pipeline_en_5.5.1_3.0_1743728675125.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("emotion_analysis_nepal_bhasa_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("emotion_analysis_nepal_bhasa_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|emotion_analysis_nepal_bhasa_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|449.1 MB|

## References

https://huggingface.co/Pavan97/emotion_analysis_new

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification