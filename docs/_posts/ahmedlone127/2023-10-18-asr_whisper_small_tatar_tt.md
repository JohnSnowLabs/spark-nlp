---
layout: model
title: Tatar asr_whisper_small_tatar WhisperForCTC from 501Good
author: John Snow Labs
name: asr_whisper_small_tatar
date: 2023-10-18
tags: [whisper, tt, open_source, asr, onnx]
task: Automatic Speech Recognition
language: tt
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
engine: onnx
annotator: WhisperForCTC
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_whisper_small_tatar` is a Tatar model originally trained by 501Good.

This model is only compatible with PySpark 3.4 and above

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_whisper_small_tatar_tt_5.1.4_3.4_1697619284915.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_whisper_small_tatar_tt_5.1.4_3.4_1697619284915.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

audioAssembler = AudioAssembler() \
    .setInputCol("audio_content") \
    .setOutputCol("audio_assembler")
    
    
speechToText  = WhisperForCTC.pretrained("asr_whisper_small_tatar","tt") \
            .setInputCols(["audio_assembler"]) \
            .setOutputCol("text")

pipeline = Pipeline().setStages([audioAssembler, speechToText])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala

val audioAssembler = new AudioAssembler() 
    .setInputCol("audio_content") 
    .setOutputCol("audio_assembler")
    
val speechToText  = WhisperForCTC.pretrained("asr_whisper_small_tatar","tt") 
            .setInputCols(Array("audio_assembler")) 
            .setOutputCol("text")

val pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_whisper_small_tatar|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|tt|
|Size:|1.7 GB|

## References

https://huggingface.co/501Good/whisper-small-tt