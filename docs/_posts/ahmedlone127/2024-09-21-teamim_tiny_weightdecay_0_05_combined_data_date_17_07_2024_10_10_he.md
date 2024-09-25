---
layout: model
title: Hebrew teamim_tiny_weightdecay_0_05_combined_data_date_17_07_2024_10_10 WhisperForCTC from cantillation
author: John Snow Labs
name: teamim_tiny_weightdecay_0_05_combined_data_date_17_07_2024_10_10
date: 2024-09-21
tags: [he, open_source, onnx, asr, whisper]
task: Automatic Speech Recognition
language: he
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

Pretrained WhisperForCTC model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`teamim_tiny_weightdecay_0_05_combined_data_date_17_07_2024_10_10` is a Hebrew model originally trained by cantillation.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/teamim_tiny_weightdecay_0_05_combined_data_date_17_07_2024_10_10_he_5.5.0_3.0_1726878969639.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/teamim_tiny_weightdecay_0_05_combined_data_date_17_07_2024_10_10_he_5.5.0_3.0_1726878969639.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
audioAssembler = AudioAssembler() \
	.setInputCol("audio_content") \
	.setOutputCol("audio_assembler")

speechToText  = WhisperForCTC.pretrained("teamim_tiny_weightdecay_0_05_combined_data_date_17_07_2024_10_10","he") \
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

val speechToText = WhisperForCTC.pretrained("teamim_tiny_weightdecay_0_05_combined_data_date_17_07_2024_10_10", "he")
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
|Model Name:|teamim_tiny_weightdecay_0_05_combined_data_date_17_07_2024_10_10|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|he|
|Size:|388.7 MB|

## References

https://huggingface.co/cantillation/Teamim-tiny_WeightDecay-0.05_Combined-Data_date-17-07-2024_10-10