---
layout: model
title: Tamil wav2vec2_large_xlsr_tamil_manandey Wav2Vec2ForCTC from manandey
author: John Snow Labs
name: wav2vec2_large_xlsr_tamil_manandey
date: 2025-04-01
tags: [ta, open_source, onnx, asr, wav2vec2]
task: Automatic Speech Recognition
language: ta
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: Wav2Vec2ForCTC
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wav2vec2_large_xlsr_tamil_manandey` is a Tamil model originally trained by manandey.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xlsr_tamil_manandey_ta_5.5.1_3.0_1743544144625.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xlsr_tamil_manandey_ta_5.5.1_3.0_1743544144625.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
audioAssembler = AudioAssembler() \
	.setInputCol("audio_content") \
	.setOutputCol("audio_assembler")

speechToText  = Wav2Vec2ForCTC.pretrained("wav2vec2_large_xlsr_tamil_manandey","ta") \
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

val speechToText = Wav2Vec2ForCTC.pretrained("wav2vec2_large_xlsr_tamil_manandey", "ta")
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
|Model Name:|wav2vec2_large_xlsr_tamil_manandey|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|ta|
|Size:|1.2 GB|

## References

https://huggingface.co/manandey/wav2vec2-large-xlsr-tamil