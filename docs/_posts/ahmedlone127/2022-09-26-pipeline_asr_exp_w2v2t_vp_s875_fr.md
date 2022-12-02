---
layout: model
title: French asr_exp_w2v2t_vp_s875 TFWav2Vec2ForCTC from jonatasgrosman
author: John Snow Labs
name: pipeline_asr_exp_w2v2t_vp_s875
date: 2022-09-26
tags: [wav2vec2, fr, audio, open_source, pipeline, asr]
task: Automatic Speech Recognition
language: fr
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2vec2  pipeline, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_exp_w2v2t_vp_s875` is a French model originally trained by jonatasgrosman.

NOTE: This pipeline only works on a CPU, if you need to use this pipeline on a GPU device please use pipeline_asr_exp_w2v2t_vp_s875_gpu

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_asr_exp_w2v2t_vp_s875_fr_4.2.0_3.0_1664186161625.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_asr_exp_w2v2t_vp_s875', lang = 'fr')
    annotations =  pipeline.transform(audioDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_asr_exp_w2v2t_vp_s875", lang = "fr")
    val annotations = pipeline.transform(audioDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_asr_exp_w2v2t_vp_s875|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|1.2 GB|

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC