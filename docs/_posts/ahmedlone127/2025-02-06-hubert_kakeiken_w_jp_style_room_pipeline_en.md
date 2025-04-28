---
layout: model
title: English hubert_kakeiken_w_jp_style_room_pipeline pipeline HubertForCTC from utakumi
author: John Snow Labs
name: hubert_kakeiken_w_jp_style_room_pipeline
date: 2025-02-06
tags: [en, open_source, pipeline, onnx]
task: Automatic Speech Recognition
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

Pretrained HubertForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hubert_kakeiken_w_jp_style_room_pipeline` is a English model originally trained by utakumi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hubert_kakeiken_w_jp_style_room_pipeline_en_5.5.1_3.0_1738862978104.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hubert_kakeiken_w_jp_style_room_pipeline_en_5.5.1_3.0_1738862978104.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hubert_kakeiken_w_jp_style_room_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hubert_kakeiken_w_jp_style_room_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hubert_kakeiken_w_jp_style_room_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|697.9 MB|

## References

https://huggingface.co/utakumi/Hubert-kakeiken-W-jp_style_room

## Included Models

- AudioAssembler
- HubertForCTC