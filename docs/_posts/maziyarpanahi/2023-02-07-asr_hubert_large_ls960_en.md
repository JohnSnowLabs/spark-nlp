---
layout: model
title: ASR HubertForCTC - asr_hubert_large_ls960
author: John Snow Labs
name: asr_hubert_large_ls960
date: 2023-02-07
tags: [open_source, hubert, audio, en, english, asr, speech, librispeech_asr, tensorflow]
task: Sentiment Analysis
language: en
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: HubertForCTC
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Hubert Model with a language modeling head on top for Connectionist Temporal Classification (CTC). 
Hubert was proposed in HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units by Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed.

The large model fine-tuned on 960h of Librispeech on 16kHz sampled speech audio. When using the model make sure that your speech input is also sampled at 16Khz.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_hubert_large_ls960_en_4.3.0_3.0_1675767067233.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_hubert_large_ls960_en_4.3.0_3.0_1675767067233.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
                
audio_assembler = AudioAssembler()\
  .setInputCol("audio_content")\
  .setOutputCol("audio_assembler")

speech_to_text = HubertForCTC.pretrained("asr_hubert_large_ls960", "en")\
  .setInputCols("audio_assembler")\
  .setOutputCol("text")

pipeline = Pipeline(stages=[
  audio_assembler,
  speech_to_text,
])

pipelineModel = pipeline.fit(audioDf)

pipelineDF = pipelineModel.transform(audioDf)
```
```scala

val audioAssembler = new AudioAssembler()
    .setInputCol("audio_content") 
    .setOutputCol("audio_assembler")

val speechToText = HubertForCTC
    .pretrained("asr_hubert_large_ls960", "en")
    .setInputCols("audio_assembler") 
    .setOutputCol("text") 

val pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

val pipelineModel = pipeline.fit(audioDf)

val pipelineDF = pipelineModel.transform(audioDf)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_hubert_large_ls960|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|en|
|Size:|1.5 GB|

## References

[https://huggingface.co/facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft)
