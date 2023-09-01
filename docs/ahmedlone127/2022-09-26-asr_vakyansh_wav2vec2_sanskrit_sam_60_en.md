---
layout: model
title: English asr_vakyansh_wav2vec2_sanskrit_sam_60 TFWav2Vec2ForCTC from Harveenchadha
author: John Snow Labs
name: asr_vakyansh_wav2vec2_sanskrit_sam_60
date: 2022-09-26
tags: [wav2vec2, en, audio, open_source, asr]
task: Automatic Speech Recognition
language: en
nav_key: models
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: Wav2Vec2ForCTC
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2vec2  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_vakyansh_wav2vec2_sanskrit_sam_60` is a English model originally trained by Harveenchadha.

NOTE: This model only works on a CPU, if you need to use this model on a GPU device please use asr_vakyansh_wav2vec2_sanskrit_sam_60_gpu

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_vakyansh_wav2vec2_sanskrit_sam_60_en_4.2.0_3.0_1664198930230.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_vakyansh_wav2vec2_sanskrit_sam_60_en_4.2.0_3.0_1664198930230.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

audio_assembler = AudioAssembler() \
    .setInputCol("audio_content") \
    .setOutputCol("audio_assembler")

speech_to_text = Wav2Vec2ForCTC \
    .pretrained("asr_vakyansh_wav2vec2_sanskrit_sam_60", "en")\
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
    .pretrained("asr_vakyansh_wav2vec2_sanskrit_sam_60", "en")
    .setInputCols("audio_assembler") 
    .setOutputCol("text") 

val pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

val pipelineModel = pipeline.fit(audioDf)

val pipelineDF = pipelineModel.transform(audioDf)

```


{:.nlu-block}
```python
import nlu
import requests
response = requests.get('https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/audio/samples/wavs/ngm_12484_01067234848.wav')
with open('ngm_12484_01067234848.wav', 'wb') as f:
    f.write(response.content)
nlu.load("en.speech2text.wav2vec2.sanskrit.v2.by_harveenchadha").predict("ngm_12484_01067234848.wav")
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_vakyansh_wav2vec2_sanskrit_sam_60|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|en|
|Size:|227.9 MB|