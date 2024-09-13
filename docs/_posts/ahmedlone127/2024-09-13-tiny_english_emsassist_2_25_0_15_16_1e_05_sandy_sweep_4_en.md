---
layout: model
title: English tiny_english_emsassist_2_25_0_15_16_1e_05_sandy_sweep_4 WhisperForCTC from saahith
author: John Snow Labs
name: tiny_english_emsassist_2_25_0_15_16_1e_05_sandy_sweep_4
date: 2024-09-13
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

Pretrained WhisperForCTC model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tiny_english_emsassist_2_25_0_15_16_1e_05_sandy_sweep_4` is a English model originally trained by saahith.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tiny_english_emsassist_2_25_0_15_16_1e_05_sandy_sweep_4_en_5.5.0_3.0_1726253991571.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tiny_english_emsassist_2_25_0_15_16_1e_05_sandy_sweep_4_en_5.5.0_3.0_1726253991571.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
audioAssembler = AudioAssembler() \
	.setInputCol("audio_content") \
	.setOutputCol("audio_assembler")

speechToText  = WhisperForCTC.pretrained("tiny_english_emsassist_2_25_0_15_16_1e_05_sandy_sweep_4","en") \
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

val speechToText = WhisperForCTC.pretrained("tiny_english_emsassist_2_25_0_15_16_1e_05_sandy_sweep_4", "en")
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
|Model Name:|tiny_english_emsassist_2_25_0_15_16_1e_05_sandy_sweep_4|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|en|
|Size:|394.7 MB|

## References

https://huggingface.co/saahith/tiny.en-EMSAssist-2-25-0.15-16-1e-05-sandy-sweep-4