---
layout: model
title: Official whisper-tiny
author: John Snow Labs
name: asr_whisper_tiny
date: 2023-10-14
tags: [whisper, en, audio, open_source, asr, xx, onnx]
task: Automatic Speech Recognition
language: xx
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
engine: onnx
annotator: WhisperForCTC
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Official pretrained Whisper model, adapted from HuggingFace transformer and curated to provide scalability and production-readiness using Spark NLP.

This is a multilingual model and supports the following languages:

Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/asr_whisper_tiny_xx_5.1.4_3.4_1697285042999.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/asr_whisper_tiny_xx_5.1.4_3.4_1697285042999.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

audioAssembler = AudioAssembler() \
    .setInputCol("audio_content") \
    .setOutputCol("audio_assembler")

speechToText = WhisperForCTC.pretrained("asr_whisper_tiny", "xx") \
    .setInputCols(["audio_assembler"]) \
    .setOutputCol("text")

pipeline = Pipeline().setStages([audioAssembler, speechToText])
processedAudioFloats = spark.createDataFrame([[rawFloats]]).toDF("audio_content")
result = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)
result.select("text.result").show(truncate = False)
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.annotators.audio.WhisperForCTC
import org.apache.spark.ml.Pipeline

val audioAssembler: AudioAssembler = new AudioAssembler()
  .setInputCol("audio_content")
  .setOutputCol("audio_assembler")

val speechToText: WhisperForCTC = WhisperForCTC
  .pretrained("asr_whisper_tiny", "xx") 
  .setInputCols("audio_assembler")
  .setOutputCol("text")

val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

val bufferedSource =
  scala.io.Source.fromFile("src/test/resources/audio/txt/librispeech_asr_0.txt")

val rawFloats = bufferedSource
  .getLines()
  .map(_.split(",").head.trim.toFloat)
  .toArray
bufferedSource.close

val processedAudioFloats = Seq(rawFloats).toDF("audio_content")

val result = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)
result.select("text.result").show(truncate = false)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|asr_whisper_tiny|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[audio_assembler]|
|Output Labels:|[text]|
|Language:|xx|
|Size:|242.8 MB|