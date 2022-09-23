---
layout: model
title: Castilian, Spanish asr_exp_w2v2r_es_vp_100k_accent_surpeninsular_8_nortepeninsular_2_s149 TFWav2Vec2ForCTC from jonatasgrosman
author: John Snow Labs
name: asr_exp_w2v2r_es_vp_100k_accent_surpeninsular_8_nortepeninsular_2_s149
date: 2022-09-23
tags: [wav2vec2, es, audio, open_source, asr]
task: Automatic Speech Recognition
language: es
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2vec2  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_exp_w2v2r_es_vp_100k_accent_surpeninsular_8_nortepeninsular_2_s149` is a Castilian, Spanish model originally trained by jonatasgrosman.

    NOTE: This model only works on a CPU, if you need to use this model on a GPU device please use asr_exp_w2v2r_es_vp_100k_accent_surpeninsular_8_nortepeninsular_2_s149_gpu

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_exp_w2v2r_es_vp_100k_accent_surpeninsular_8_nortepeninsular_2_s149_es_4.2.0_3.0_1663948449296.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    audio_assembler = AudioAssembler() \
        .setInputCol("audio_content") \
        .setOutputCol("audio_assembler")

    speech_to_text = Wav2Vec2ForCTC \
        .pretrained("asr_exp_w2v2r_es_vp_100k_accent_surpeninsular_8_nortepeninsular_2_s149", "es")\
        .setInputCols("audio_assembler") \
        .setOutputCol("text")

    pipeline = Pipeline(stages=[
      audio_assembler,
      speech_to_text,
    ])

    pipelineModel = pipeline.fit(audioDf)

    pipelineDF = pipelineModel.transform(audioDf)
```
```scala

    val audioAssembler = new AudioAssembler()
        .setInputCol("audio_content") 
        .setOutputCol("audio_assembler")

    val speechToText = Wav2Vec2ForCTC
        .pretrained("asr_exp_w2v2r_es_vp_100k_accent_surpeninsular_8_nortepeninsular_2_s149", "es")
        .setInputCols("audio_assembler") 
        .setOutputCol("text") 

    val pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

    val pipelineModel = pipeline.fit(audioDf)

    val pipelineDF = pipelineModel.transform(audioDf)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_exp_w2v2r_es_vp_100k_accent_surpeninsular_8_nortepeninsular_2_s149|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|es|
|Size:|1.2 GB|