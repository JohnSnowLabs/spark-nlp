---
layout: model
title: English title_generation1_for_yt_videos_pipeline pipeline T5Transformer from michc
author: John Snow Labs
name: title_generation1_for_yt_videos_pipeline
date: 2025-01-26
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`title_generation1_for_yt_videos_pipeline` is a English model originally trained by michc.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/title_generation1_for_yt_videos_pipeline_en_5.5.1_3.0_1737850576143.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/title_generation1_for_yt_videos_pipeline_en_5.5.1_3.0_1737850576143.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("title_generation1_for_yt_videos_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("title_generation1_for_yt_videos_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|title_generation1_for_yt_videos_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.0 GB|

## References

https://huggingface.co/michc/title-generation1-for-YT-videos

## Included Models

- DocumentAssembler
- T5Transformer