---
layout: model
title: Castilian, Spanish openai_whisper_base_spanish_ecu911_pasobajo WhisperForCTC from DanielMarquez
author: John Snow Labs
name: openai_whisper_base_spanish_ecu911_pasobajo
date: 2024-09-19
tags: [es, open_source, onnx, asr, whisper]
task: Automatic Speech Recognition
language: es
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: WhisperForCTC
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`openai_whisper_base_spanish_ecu911_pasobajo` is a Castilian, Spanish model originally trained by DanielMarquez.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/openai_whisper_base_spanish_ecu911_pasobajo_es_5.5.0_3.0_1726714409500.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/openai_whisper_base_spanish_ecu911_pasobajo_es_5.5.0_3.0_1726714409500.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
audioAssembler = AudioAssembler() \
	.setInputCol("audio_content") \
	.setOutputCol("audio_assembler")

speechToText  = WhisperForCTC.pretrained("openai_whisper_base_spanish_ecu911_pasobajo","es") \
     .setInputCols(["audio_assembler"]) \
     .setOutputCol("text")

pipeline = Pipeline().setStages([audioAssembler, speechToText])
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val audioAssembler = new DocumentAssembler()
    .setInputCols("audio_content")
    .setOutputCols("audio_assembler")

val speechToText = WhisperForCTC.pretrained("openai_whisper_base_spanish_ecu911_pasobajo", "es")
    .setInputCols(Array("audio_assembler")) 
    .setOutputCol("text") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, speechToText))
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|openai_whisper_base_spanish_ecu911_pasobajo|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|es|
|Size:|615.7 MB|

## References

https://huggingface.co/DanielMarquez/openai-whisper-base-es_ecu911-PasoBajo