---
layout: model
title: English distil_asr_whisper_mediumWhisperForCTC from distil-whisper
author: John Snow Labs
name: distil_asr_whisper_medium
date: 2024-02-25
tags: [whisper, en, open_source, onnx]
task: Automatic Speech Recognition
language: en
edition: Spark NLP 5.2.4
spark_version: 3.4
supported: true
engine: onnx
annotator: WhisperForCTC
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.distil_asr_whisper_medium is a English model originally trained by distil-whisper.

This model is only compatible with PySpark 3.4 and above

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distil_asr_whisper_medium_en_5.2.4_3.4_1708901703317.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distil_asr_whisper_medium_en_5.2.4_3.4_1708901703317.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
audioAssembler = AudioAssembler() \
    .setInputCol("audio_content") \
    .setOutputCol("audio_assembler")


speechToText  = WhisperForCTC.pretrained("distil_asr_whisper_medium","en") \
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
    
val speechToText  = WhisperForCTC.pretrained("distil_asr_whisper_medium","en") 
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
|Model Name:|distil_asr_whisper_medium|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|en|
|Size:|1.4 GB|

## References

https://huggingface.co/distil-whisper/distil-medium.en