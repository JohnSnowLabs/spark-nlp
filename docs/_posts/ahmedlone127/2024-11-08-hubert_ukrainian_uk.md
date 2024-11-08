---
layout: model
title: Ukrainian hubert_ukrainian HubertForCTC from Yehor
author: John Snow Labs
name: hubert_ukrainian
date: 2024-11-08
tags: [uk, open_source, onnx, asr, hubert]
task: Automatic Speech Recognition
language: uk
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: HubertForCTC
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained HubertForCTC model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hubert_ukrainian` is a Ukrainian model originally trained by Yehor.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hubert_ukrainian_uk_5.5.1_3.0_1731106423734.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hubert_ukrainian_uk_5.5.1_3.0_1731106423734.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
audioAssembler = AudioAssembler() \
	.setInputCol("audio_content") \
	.setOutputCol("audio_assembler")

speechToText  = HubertForCTC.pretrained("hubert_ukrainian","uk") \
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

val speechToText = HubertForCTC.pretrained("hubert_ukrainian", "uk")
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
|Model Name:|hubert_ukrainian|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|uk|
|Size:|708.6 MB|

## References

https://huggingface.co/Yehor/hubert-uk