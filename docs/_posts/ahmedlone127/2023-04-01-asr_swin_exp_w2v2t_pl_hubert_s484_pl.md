---
layout: model
title: Polish asr_swin_exp_w2v2t_pl_hubert_s484 TFHubertForCTC from jonatasgrosman
author: John Snow Labs
name: asr_swin_exp_w2v2t_pl_hubert_s484
date: 2023-04-01
tags: [hubert, pl, open_source, audio, asr, tensorflow]
task: Automatic Speech Recognition
language: pl
edition: Spark NLP 4.4.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: HubertForCTC
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Hubert  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`asr_swin_exp_w2v2t_pl_hubert_s484` is a Polish model originally trained by jonatasgrosman.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_swin_exp_w2v2t_pl_hubert_s484_pl_4.4.0_3.0_1680361854210.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_swin_exp_w2v2t_pl_hubert_s484_pl_4.4.0_3.0_1680361854210.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

audio_assembler = AudioAssembler() \
    .setInputCol("audio_content") \
    .setOutputCol("audio_assembler")

speech_to_text = HubertForCTC \
    .pretrained("asr_swin_exp_w2v2t_pl_hubert_s484", "pl")\
    .setInputCols("audio_assembler") \
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
    .pretrained("asr_swin_exp_w2v2t_pl_hubert_s484", "pl")
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
|Model Name:|asr_swin_exp_w2v2t_pl_hubert_s484|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|pl|
|Size:|2.4 GB|