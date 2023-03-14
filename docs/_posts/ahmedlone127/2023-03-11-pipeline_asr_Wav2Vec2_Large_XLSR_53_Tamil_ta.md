---
layout: model
title: Tamil asr_Wav2Vec2_Large_XLSR_53_Tamil TFWav2Vec2ForCTC from infinitejoy
author: John Snow Labs
name: pipeline_asr_Wav2Vec2_Large_XLSR_53_Tamil
date: 2023-03-11
tags: [wav2vec2, ta, audio, open_source, pipeline, asr]
task: Automatic Speech Recognition
language: ta
edition: Spark NLP 4.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2vec2  pipeline, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_Wav2Vec2_Large_XLSR_53_Tamil` is a Tamil model originally trained by infinitejoy.

NOTE: This pipeline only works on a CPU, if you need to use this pipeline on a GPU device please use pipeline_asr_Wav2Vec2_Large_XLSR_53_Tamil_gpu

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_asr_Wav2Vec2_Large_XLSR_53_Tamil_ta_4.4.0_3.0_1678499422150.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_asr_Wav2Vec2_Large_XLSR_53_Tamil_ta_4.4.0_3.0_1678499422150.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_asr_Wav2Vec2_Large_XLSR_53_Tamil', lang = 'ta')
    annotations =  pipeline.transform(audioDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_asr_Wav2Vec2_Large_XLSR_53_Tamil", lang = "ta")
    val annotations = pipeline.transform(audioDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_asr_Wav2Vec2_Large_XLSR_53_Tamil|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ta|
|Size:|1.2 GB|

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC