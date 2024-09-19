---
layout: model
title: English workstation_whisper_base_finetune_teacher__babble_noise_mozilla_100_epochs_batch_4 WhisperForCTC from rohitp1
author: John Snow Labs
name: workstation_whisper_base_finetune_teacher__babble_noise_mozilla_100_epochs_batch_4
date: 2024-09-17
tags: [en, open_source, onnx, asr, whisper]
task: Automatic Speech Recognition
language: en
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

Pretrained WhisperForCTC model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`workstation_whisper_base_finetune_teacher__babble_noise_mozilla_100_epochs_batch_4` is a English model originally trained by rohitp1.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/workstation_whisper_base_finetune_teacher__babble_noise_mozilla_100_epochs_batch_4_en_5.5.0_3.0_1726546114115.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/workstation_whisper_base_finetune_teacher__babble_noise_mozilla_100_epochs_batch_4_en_5.5.0_3.0_1726546114115.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
audioAssembler = AudioAssembler() \
	.setInputCol("audio_content") \
	.setOutputCol("audio_assembler")

speechToText  = WhisperForCTC.pretrained("workstation_whisper_base_finetune_teacher__babble_noise_mozilla_100_epochs_batch_4","en") \
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

val speechToText = WhisperForCTC.pretrained("workstation_whisper_base_finetune_teacher__babble_noise_mozilla_100_epochs_batch_4", "en")
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
|Model Name:|workstation_whisper_base_finetune_teacher__babble_noise_mozilla_100_epochs_batch_4|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|en|
|Size:|646.7 MB|

## References

https://huggingface.co/rohitp1/workstation_whisper_base_finetune_teacher__babble_noise_mozilla_100_epochs_batch_4