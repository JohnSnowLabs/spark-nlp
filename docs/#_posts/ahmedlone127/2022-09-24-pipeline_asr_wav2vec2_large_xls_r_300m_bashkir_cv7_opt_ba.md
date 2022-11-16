---
layout: model
title: Bashkir asr_wav2vec2_large_xls_r_300m_bashkir_cv7_opt TFWav2Vec2ForCTC from AigizK
author: John Snow Labs
name: pipeline_asr_wav2vec2_large_xls_r_300m_bashkir_cv7_opt
date: 2022-09-24
tags: [wav2vec2, ba, audio, open_source, pipeline, asr]
task: Automatic Speech Recognition
language: ba
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2vec2  pipeline, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_wav2vec2_large_xls_r_300m_bashkir_cv7_opt` is a Bashkir model originally trained by AigizK.

NOTE: This pipeline only works on a CPU, if you need to use this pipeline on a GPU device please use pipeline_asr_wav2vec2_large_xls_r_300m_bashkir_cv7_opt_gpu

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_asr_wav2vec2_large_xls_r_300m_bashkir_cv7_opt_ba_4.2.0_3.0_1664040387674.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_asr_wav2vec2_large_xls_r_300m_bashkir_cv7_opt', lang = 'ba')
    annotations =  pipeline.transform(audioDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_asr_wav2vec2_large_xls_r_300m_bashkir_cv7_opt", lang = "ba")
    val annotations = pipeline.transform(audioDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_asr_wav2vec2_large_xls_r_300m_bashkir_cv7_opt|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ba|
|Size:|1.2 GB|

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC