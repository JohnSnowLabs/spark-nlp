---
layout: model
title: Western Frisian whisper_small_western_frisian_dutch_transfer_from_english_polixonrio WhisperForCTC from polixonrio
author: John Snow Labs
name: whisper_small_western_frisian_dutch_transfer_from_english_polixonrio
date: 2024-09-16
tags: [fy, open_source, onnx, asr, whisper]
task: Automatic Speech Recognition
language: fy
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

Pretrained WhisperForCTC model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_small_western_frisian_dutch_transfer_from_english_polixonrio` is a Western Frisian model originally trained by polixonrio.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_small_western_frisian_dutch_transfer_from_english_polixonrio_fy_5.5.0_3.0_1726485114236.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_small_western_frisian_dutch_transfer_from_english_polixonrio_fy_5.5.0_3.0_1726485114236.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
audioAssembler = AudioAssembler() \
	.setInputCol("audio_content") \
	.setOutputCol("audio_assembler")

speechToText  = WhisperForCTC.pretrained("whisper_small_western_frisian_dutch_transfer_from_english_polixonrio","fy") \
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

val speechToText = WhisperForCTC.pretrained("whisper_small_western_frisian_dutch_transfer_from_english_polixonrio", "fy")
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
|Model Name:|whisper_small_western_frisian_dutch_transfer_from_english_polixonrio|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|fy|
|Size:|1.7 GB|

## References

https://huggingface.co/polixonrio/whisper-small-fy-NL-Transfer-From-EN