---
layout: model
title: English dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8 WhisperForCTC from rohitp1
author: John Snow Labs
name: dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8
date: 2024-09-24
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

Pretrained WhisperForCTC model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8` is a English model originally trained by rohitp1.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8_en_5.5.0_3.0_1727146591942.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8_en_5.5.0_3.0_1727146591942.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
audioAssembler = AudioAssembler() \
	.setInputCol("audio_content") \
	.setOutputCol("audio_assembler")

speechToText  = WhisperForCTC.pretrained("dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8","en") \
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

val speechToText = WhisperForCTC.pretrained("dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8", "en")
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
|Model Name:|dgx1_whisper_base_finetune_teacher_norwegian_noise_mozilla_100_epochs_batch_8|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|en|
|Size:|646.8 MB|

## References

https://huggingface.co/rohitp1/dgx1_whisper_base_finetune_teacher_no_noise_mozilla_100_epochs_batch_8