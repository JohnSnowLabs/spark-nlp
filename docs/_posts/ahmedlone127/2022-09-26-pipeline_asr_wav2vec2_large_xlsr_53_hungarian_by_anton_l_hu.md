---
layout: model
title: Hungarian asr_wav2vec2_large_xlsr_53_hungarian_by_anton_l TFWav2Vec2ForCTC from anton-l
author: John Snow Labs
name: pipeline_asr_wav2vec2_large_xlsr_53_hungarian_by_anton_l
date: 2022-09-26
tags: [wav2vec2, hu, audio, open_source, pipeline, asr]
task: Automatic Speech Recognition
language: hu
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2vec2  pipeline, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_wav2vec2_large_xlsr_53_hungarian_by_anton_l` is a Hungarian model originally trained by anton-l.

NOTE: This pipeline only works on a CPU, if you need to use this pipeline on a GPU device please use pipeline_asr_wav2vec2_large_xlsr_53_hungarian_by_anton_l_gpu

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_asr_wav2vec2_large_xlsr_53_hungarian_by_anton_l_hu_4.2.0_3.0_1664203696815.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_asr_wav2vec2_large_xlsr_53_hungarian_by_anton_l', lang = 'hu')
    annotations =  pipeline.transform(audioDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_asr_wav2vec2_large_xlsr_53_hungarian_by_anton_l", lang = "hu")
    val annotations = pipeline.transform(audioDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_asr_wav2vec2_large_xlsr_53_hungarian_by_anton_l|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|hu|
|Size:|1.2 GB|

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC