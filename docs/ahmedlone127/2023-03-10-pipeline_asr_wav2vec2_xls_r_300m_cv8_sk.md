---
layout: model
title: Slovak asr_wav2vec2_xls_r_300m_cv8 TFWav2Vec2ForCTC from comodoro
author: John Snow Labs
name: pipeline_asr_wav2vec2_xls_r_300m_cv8
date: 2023-03-10
tags: [wav2vec2, sk, audio, open_source, pipeline, asr]
task: Automatic Speech Recognition
language: sk
edition: Spark NLP 4.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2vec2  pipeline, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_wav2vec2_xls_r_300m_cv8` is a Slovak model originally trained by comodoro.

NOTE: This pipeline only works on a CPU, if you need to use this pipeline on a GPU device please use pipeline_asr_wav2vec2_xls_r_300m_cv8_gpu

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_asr_wav2vec2_xls_r_300m_cv8_sk_4.4.0_3.0_1678478961259.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pipeline_asr_wav2vec2_xls_r_300m_cv8_sk_4.4.0_3.0_1678478961259.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_asr_wav2vec2_xls_r_300m_cv8', lang = 'sk')
    annotations =  pipeline.transform(audioDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_asr_wav2vec2_xls_r_300m_cv8", lang = "sk")
    val annotations = pipeline.transform(audioDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_asr_wav2vec2_xls_r_300m_cv8|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sk|
|Size:|1.2 GB|

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC