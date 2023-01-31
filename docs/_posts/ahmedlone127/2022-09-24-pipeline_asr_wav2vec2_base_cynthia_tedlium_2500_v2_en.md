---
layout: model
title: English asr_wav2vec2_base_cynthia_tedlium_2500_v2 TFWav2Vec2ForCTC from huyue012
author: John Snow Labs
name: pipeline_asr_wav2vec2_base_cynthia_tedlium_2500_v2
date: 2022-09-24
tags: [wav2vec2, en, audio, open_source, pipeline, asr]
task: Automatic Speech Recognition
language: en
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2vec2  pipeline, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_wav2vec2_base_cynthia_tedlium_2500_v2` is a English model originally trained by huyue012.

NOTE: This pipeline only works on a CPU, if you need to use this pipeline on a GPU device please use pipeline_asr_wav2vec2_base_cynthia_tedlium_2500_v2_gpu

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_asr_wav2vec2_base_cynthia_tedlium_2500_v2_en_4.2.0_3.0_1664040558353.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_asr_wav2vec2_base_cynthia_tedlium_2500_v2_en_4.2.0_3.0_1664040558353.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_asr_wav2vec2_base_cynthia_tedlium_2500_v2', lang = 'en')
    annotations =  pipeline.transform(audioDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_asr_wav2vec2_base_cynthia_tedlium_2500_v2", lang = "en")
    val annotations = pipeline.transform(audioDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_asr_wav2vec2_base_cynthia_tedlium_2500_v2|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|349.2 MB|

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC