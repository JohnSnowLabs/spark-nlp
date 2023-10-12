---
layout: model
title: Multilingual asuri_whisper_tiny WhisperForCTC from openai
author: John Snow Labs
name: asuri_whisper_tiny
date: 2023-10-12
tags: [whisper, xx, open_source, asr]
task: Automatic Speech Recognition
language: xx
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asuri_whisper_tiny` is a Multilingual model originally trained by openai.

This model is only compatible with PySpark 3.4 and above

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asuri_whisper_tiny_xx_5.1.4_3.4_1697146106142.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asuri_whisper_tiny_xx_5.1.4_3.4_1697146106142.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


audioAssembler = AudioAssembler() \
    .setInputCol("audio_content") \
    .setOutputCol("audio_assembler")
    
    
speechToText  = WhisperForCTC.pretrained("asuri_whisper_tiny","xx") \
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
    
val speechToText  = WhisperForCTC.pretrained("asuri_whisper_tiny","xx") 
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
|Model Name:|asuri_whisper_tiny|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|242.8 MB|

## References

https://huggingface.co/openai/whisper-tiny

## Included Models

- AudioAssembler
- WhisperForCTC