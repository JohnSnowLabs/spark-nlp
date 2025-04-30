---
layout: model
title: English seq_xls_r_fleurs_zulu_run3_asr_xhosa_run8_pipeline pipeline Wav2Vec2ForCTC from lucas-meyer
author: John Snow Labs
name: seq_xls_r_fleurs_zulu_run3_asr_xhosa_run8_pipeline
date: 2025-04-06
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

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`seq_xls_r_fleurs_zulu_run3_asr_xhosa_run8_pipeline` is a English model originally trained by lucas-meyer.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/seq_xls_r_fleurs_zulu_run3_asr_xhosa_run8_pipeline_en_5.5.1_3.0_1743934354127.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/seq_xls_r_fleurs_zulu_run3_asr_xhosa_run8_pipeline_en_5.5.1_3.0_1743934354127.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("seq_xls_r_fleurs_zulu_run3_asr_xhosa_run8_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("seq_xls_r_fleurs_zulu_run3_asr_xhosa_run8_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|seq_xls_r_fleurs_zulu_run3_asr_xhosa_run8_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.2 GB|

## References

https://huggingface.co/lucas-meyer/seq-xls-r-fleurs_zu-run3-asr_xh-run8

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC