---
layout: model
title: English speech_chatgpt_base_nar_v2_epoch4_wotrans_scratch_pipeline pipeline BartTransformer from lca0503
author: John Snow Labs
name: speech_chatgpt_base_nar_v2_epoch4_wotrans_scratch_pipeline
date: 2025-04-04
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

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`speech_chatgpt_base_nar_v2_epoch4_wotrans_scratch_pipeline` is a English model originally trained by lca0503.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/speech_chatgpt_base_nar_v2_epoch4_wotrans_scratch_pipeline_en_5.5.1_3.0_1743735995531.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/speech_chatgpt_base_nar_v2_epoch4_wotrans_scratch_pipeline_en_5.5.1_3.0_1743735995531.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("speech_chatgpt_base_nar_v2_epoch4_wotrans_scratch_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("speech_chatgpt_base_nar_v2_epoch4_wotrans_scratch_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|speech_chatgpt_base_nar_v2_epoch4_wotrans_scratch_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|882.4 MB|

## References

https://huggingface.co/lca0503/speech-chatgpt-base-nar-v2-epoch4-wotrans-scratch

## Included Models

- DocumentAssembler
- BartTransformer