---
layout: model
title: Dutch, Flemish fifa_wc22_winner_language_model WhisperForCTC from AlexMo
author: John Snow Labs
name: fifa_wc22_winner_language_model
date: 2024-09-16
tags: [nl, open_source, onnx, asr, whisper]
task: Automatic Speech Recognition
language: nl
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

Pretrained WhisperForCTC model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fifa_wc22_winner_language_model` is a Dutch, Flemish model originally trained by AlexMo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fifa_wc22_winner_language_model_nl_5.5.0_3.0_1726480217648.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fifa_wc22_winner_language_model_nl_5.5.0_3.0_1726480217648.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
audioAssembler = AudioAssembler() \
	.setInputCol("audio_content") \
	.setOutputCol("audio_assembler")

speechToText  = WhisperForCTC.pretrained("fifa_wc22_winner_language_model","nl") \
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

val speechToText = WhisperForCTC.pretrained("fifa_wc22_winner_language_model", "nl")
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
|Model Name:|fifa_wc22_winner_language_model|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|nl|
|Size:|1.7 GB|

## References

https://huggingface.co/AlexMo/FIFA_WC22_WINNER_LANGUAGE_MODEL