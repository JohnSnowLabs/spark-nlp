---
layout: model
title: English thai_lyrics_wangchan_pipeline pipeline CamemBertForSequenceClassification from Nitcha
author: John Snow Labs
name: thai_lyrics_wangchan_pipeline
date: 2025-01-25
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

Pretrained CamemBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`thai_lyrics_wangchan_pipeline` is a English model originally trained by Nitcha.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/thai_lyrics_wangchan_pipeline_en_5.5.1_3.0_1737823349108.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/thai_lyrics_wangchan_pipeline_en_5.5.1_3.0_1737823349108.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("thai_lyrics_wangchan_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("thai_lyrics_wangchan_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|thai_lyrics_wangchan_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|394.4 MB|

## References

https://huggingface.co/Nitcha/thai-lyrics-wangchan

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForSequenceClassification