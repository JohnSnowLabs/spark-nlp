---
layout: model
title: Swedish swedish_fine_tuned_whisper_model_nadiaholmlund WhisperForCTC from NadiaHolmlund
author: John Snow Labs
name: swedish_fine_tuned_whisper_model_nadiaholmlund
date: 2024-09-17
tags: [sv, open_source, onnx, asr, whisper]
task: Automatic Speech Recognition
language: sv
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

Pretrained WhisperForCTC model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`swedish_fine_tuned_whisper_model_nadiaholmlund` is a Swedish model originally trained by NadiaHolmlund.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/swedish_fine_tuned_whisper_model_nadiaholmlund_sv_5.5.0_3.0_1726563743022.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/swedish_fine_tuned_whisper_model_nadiaholmlund_sv_5.5.0_3.0_1726563743022.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
audioAssembler = AudioAssembler() \
	.setInputCol("audio_content") \
	.setOutputCol("audio_assembler")

speechToText  = WhisperForCTC.pretrained("swedish_fine_tuned_whisper_model_nadiaholmlund","sv") \
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

val speechToText = WhisperForCTC.pretrained("swedish_fine_tuned_whisper_model_nadiaholmlund", "sv")
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
|Model Name:|swedish_fine_tuned_whisper_model_nadiaholmlund|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|sv|
|Size:|390.9 MB|

## References

https://huggingface.co/NadiaHolmlund/Swedish_Fine_Tuned_Whisper_Model