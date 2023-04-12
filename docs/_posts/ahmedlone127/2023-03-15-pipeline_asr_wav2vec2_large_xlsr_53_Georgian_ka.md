---
layout: model
title: Georgian asr_wav2vec2_large_xlsr_53_Georgian TFWav2Vec2ForCTC from MehdiHosseiniMoghadam
author: John Snow Labs
name: pipeline_asr_wav2vec2_large_xlsr_53_Georgian
date: 2023-03-15
tags: [wav2vec2, ka, audio, open_source, pipeline, asr]
task: Automatic Speech Recognition
language: ka
edition: Spark NLP 4.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2vec2  pipeline, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_wav2vec2_large_xlsr_53_Georgian` is a Georgian model originally trained by MehdiHosseiniMoghadam.

NOTE: This pipeline only works on a CPU, if you need to use this pipeline on a GPU device please use pipeline_asr_wav2vec2_large_xlsr_53_Georgian_gpu

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_asr_wav2vec2_large_xlsr_53_Georgian_ka_4.4.0_3.0_1678908247126.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_asr_wav2vec2_large_xlsr_53_Georgian_ka_4.4.0_3.0_1678908247126.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_asr_wav2vec2_large_xlsr_53_Georgian', lang = 'ka')
    annotations =  pipeline.transform(audioDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_asr_wav2vec2_large_xlsr_53_Georgian", lang = "ka")
    val annotations = pipeline.transform(audioDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_asr_wav2vec2_large_xlsr_53_Georgian|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ka|
|Size:|1.2 GB|

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC