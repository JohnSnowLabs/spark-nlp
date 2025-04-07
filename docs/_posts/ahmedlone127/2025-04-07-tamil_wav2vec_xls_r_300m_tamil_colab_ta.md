---
layout: model
title: Tamil tamil_wav2vec_xls_r_300m_tamil_colab Wav2Vec2ForCTC from bharat-raghunathan
author: John Snow Labs
name: tamil_wav2vec_xls_r_300m_tamil_colab
date: 2025-04-07
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

Pretrained Wav2Vec2ForCTC model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tamil_wav2vec_xls_r_300m_tamil_colab` is a Tamil model originally trained by bharat-raghunathan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tamil_wav2vec_xls_r_300m_tamil_colab_ta_5.5.1_3.0_1744021674002.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tamil_wav2vec_xls_r_300m_tamil_colab_ta_5.5.1_3.0_1744021674002.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
audioAssembler = AudioAssembler() \
	.setInputCol("audio_content") \
	.setOutputCol("audio_assembler")

speechToText  = Wav2Vec2ForCTC.pretrained("tamil_wav2vec_xls_r_300m_tamil_colab","ta") \
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

val speechToText = Wav2Vec2ForCTC.pretrained("tamil_wav2vec_xls_r_300m_tamil_colab", "ta")
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
|Model Name:|tamil_wav2vec_xls_r_300m_tamil_colab|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|ta|
|Size:|1.2 GB|

## References

https://huggingface.co/bharat-raghunathan/Tamil-Wav2Vec-xls-r-300m-Tamil-colab