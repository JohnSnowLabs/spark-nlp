---
layout: model
title: English asr_wav2vec2_base_960h TFWav2Vec2ForCTC from facebook
author: John Snow Labs
name: asr_wav2vec2_base_960h
date: 2024-08-11
tags: [wav2vec2, en, open_source, onnx]
task: Automatic Speech Recognition
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: Wav2Vec2ForCTC
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

“
        
        
        Pretrained Wav2vec2 model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.asr_wav2vec2_base_960h_by_facebook is a English model originally trained by facebook.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_wav2vec2_base_960h_en_5.4.2_3.0_1723388189937.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_wav2vec2_base_960h_en_5.4.2_3.0_1723388189937.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

audio_assembler = AudioAssembler() \
    .setInputCol("audio_content") \
    .setOutputCol("audio_assembler")

speech_to_text = Wav2Vec2ForCTC \
    .pretrained("asr_wav2vec2_base_960h", "en")\
    .setInputCols("audio_assembler") \
    .setOutputCol("text")


```
```scala

val audioAssembler = new AudioAssembler()
    .setInputCol("audio_content") 
    .setOutputCol("audio_assembler")

val speechToText = Wav2Vec2ForCTC
    .pretrained("asr_wav2vec2_base_960h", "en")
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
|Model Name:|asr_wav2vec2_base_960h|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|en|
|Size:|233.0 MB|