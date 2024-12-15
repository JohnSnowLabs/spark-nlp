---
layout: model
title: Tagalog tagalog_hatespeech_detection_pipeline pipeline RoBertaForSequenceClassification from ggpt1006
author: John Snow Labs
name: tagalog_hatespeech_detection_pipeline
date: 2024-12-14
tags: [tl, open_source, pipeline, onnx]
task: Text Classification
language: tl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tagalog_hatespeech_detection_pipeline` is a Tagalog model originally trained by ggpt1006.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tagalog_hatespeech_detection_pipeline_tl_5.5.1_3.0_1734218060883.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tagalog_hatespeech_detection_pipeline_tl_5.5.1_3.0_1734218060883.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tagalog_hatespeech_detection_pipeline", lang = "tl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tagalog_hatespeech_detection_pipeline", lang = "tl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tagalog_hatespeech_detection_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|tl|
|Size:|442.7 MB|

## References

https://huggingface.co/ggpt1006/tl-hatespeech-detection

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification