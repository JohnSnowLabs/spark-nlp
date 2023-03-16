---
layout: model
title: Dhivehi, Divehi, Maldivian asr_wav2vec2_xls_r_1b TFWav2Vec2ForCTC from shahukareem
author: John Snow Labs
name: asr_wav2vec2_xls_r_1b
date: 2023-03-16
tags: [wav2vec2, dv, audio, open_source, asr, tensorflow]
task: Automatic Speech Recognition
language: dv
edition: Spark NLP 4.4.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: Wav2Vec2ForCTC
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2vec2  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_wav2vec2_xls_r_1b` is a Dhivehi, Divehi, Maldivian model originally trained by shahukareem.

NOTE: This model only works on a CPU, if you need to use this model on a GPU device please use asr_wav2vec2_xls_r_1b_gpu

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_wav2vec2_xls_r_1b_dv_4.4.0_3.0_1679000839219.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_wav2vec2_xls_r_1b_dv_4.4.0_3.0_1679000839219.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

audio_assembler = AudioAssembler() \
    .setInputCol("audio_content") \
    .setOutputCol("audio_assembler")

speech_to_text = Wav2Vec2ForCTC \
    .pretrained("asr_wav2vec2_xls_r_1b", "dv")\
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
    .pretrained("asr_wav2vec2_xls_r_1b", "dv")
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
|Model Name:|asr_wav2vec2_xls_r_1b|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|dv|
|Size:|3.6 GB|