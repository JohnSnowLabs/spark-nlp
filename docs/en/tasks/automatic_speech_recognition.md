---
layout: docs
header: true
seotitle:
title: Automatic Speech Recognition
permalink: docs/en/tasks/automatic_speech_recognition
key: docs-tasks-automatic-speech-recognition
modify_date: "2024-09-26"
show_nav: true
sidebar:
  nav: sparknlp
---

Automatic Speech Recognition (ASR), also known as Speech-to-Text (STT), is the process of converting spoken language into written text. ASR systems analyze audio signals to identify speech patterns and map them to words using models that combine acoustic, linguistic, and contextual understanding. This enables devices and applications to interpret human speech naturally and accurately.

ASR is widely used in virtual assistants like **Siri**, **Alexa**, and **Google Assistant** for voice commands, and it powers automatic captioning, meeting transcriptions, and accessibility tools that help users with hearing impairments. By turning speech into text, ASR enables faster, hands-free interaction and improves access to spoken content across many applications.

<div style="text-align: center;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/hscJK-4kA_A?si=xYgz6ejc-2XvXcoR" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

## Picking a Model

When picking a model for Automatic Speech Recognition, consider the language coverage, accuracy requirements, and computational resources available. Pretrained models like **Whisper** (by OpenAI) are excellent general-purpose options that support over 100 languages and handle accents, background noise, and different speech styles effectively. For English-only or smaller-scale use cases, models such as **Wav2Vec 2.0**, **Conformer**, or **DeepSpeech** offer strong accuracy with lower resource demands.

If your application involves multiple languages, look for **multilingual ASR models** that can automatically detect and transcribe speech in different languages within the same audio. For real-time or on-device applications, lightweight versions of these models are better suited due to their faster processing speed. Ultimately, the best model depends on your trade-off between **accuracy, speed, language support, and deployment constraints**.

To explore and select from a variety of models, visit [Spark NLP Models](https://sparknlp.org/models)

#### Recommended Models for Automatic Speech Recognition Tasks
- **General Speech Recognition:** Use models like [`asr_wav2vec2_large_xlsr_53_english_by_jonatasgrosman`](https://sparknlp.org/2022/09/24/asr_wav2vec2_large_xlsr_53_english_by_jonatasgrosman_en.html){:target="_blank"} for general-purpose transcription.
- **Multilingual Support:** For applications requiring support for multiple languages, consider using models like [`asr_wav2vec2_large_xlsr_53_portuguese_by_jonatasgrosman`](https://sparknlp.org/2021/12/15/wav2vec2.html){:target="_blank"} from the [`Wav2Vec2ForCTC`](https://sparknlp.org/docs/en/transformers#wav2vec2forctc){:target="_blank"} transformer.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

audioAssembler = AudioAssembler() \
    .setInputCol("audio_content") \
    .setOutputCol("audio_assembler")

speechToText = Wav2Vec2ForCTC \
    .pretrained() \
    .setInputCols(["audio_assembler"]) \
    .setOutputCol("text")

pipeline = Pipeline().setStages([
  audioAssembler, 
  speechToText
])

processedAudioFloats = spark.createDataFrame([[rawFloats]]).toDF("audio_content")

model = pipeline.fit(processedAudioFloats)
result = model.transform(processedAudioFloats)

result.select("text.result").show(truncate=False)

```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types._

val audioAssembler = new AudioAssembler()
  .setInputCol("audio_content")
  .setOutputCol("audio_assembler")

val speechToText = Wav2Vec2ForCTC
  .pretrained()
  .setInputCols("audio_assembler")
  .setOutputCol("text")

val pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

val schema = StructType(Array(StructField("audio_content", ArrayType(FloatType))))
val data = Seq(Seq(rawFloats))
val processedAudioFloats = spark.createDataFrame(data.map(Tuple1(_))).toDF("audio_content")

val model = pipeline.fit(processedAudioFloats)
val result = model.transform(processedAudioFloats)

result.select("text.result").show(false)

```
</div>

<div class="tabs-box" markdown="1">
```
+------------------------------------------------------------------------------------------+
|result                                                                                    |
+------------------------------------------------------------------------------------------+
|[MISTER QUILTER IS THE APOSTLE OF THE MIDLE CLASES AND WE ARE GLAD TO WELCOME HIS GOSPEL ]|
+------------------------------------------------------------------------------------------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of ASR models in real time, visit our interactive demos:

- **[Wav2Vec2ForCTC](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-Wav2Vec2ForCTC){:target="_blank"}**
- **[WhisperForCTC](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-WhisperForCTC){:target="_blank"}**
- **[HubertForCTC](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-HubertForCTC){:target="_blank"}**

## Useful Resources

Want to dive deeper into Automatic Speech Recognition with Spark NLP? Here are some curated resources to help you get started and explore further:

**Articles and Guides**
- *[Converting Speech to Text with Spark NLP and Python](https://www.johnsnowlabs.com/converting-speech-to-text-with-spark-nlp-and-python/){:target="_blank"}*
- *[Simplify Your Speech Recognition Workflow with SparkNLP](https://medium.com/spark-nlp/simplify-your-speech-recognition-workflow-with-sparknlp-e381606e4e82){:target="_blank"}*
- *[Vision Transformers and Automatic Speech Recognition in Spark NLP](https://www.nlpsummit.org/vision-transformers-and-automatic-speech-recognition-in-spark-nlp/){:target="_blank"}*

**Notebooks**
- *[Automatic Speech Recognition in Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/17.Speech_Recognition.ipynb){:target="_blank"}*
- *[Speech Recognition Transformers in Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/annotation/audio){:target="_blank"}*
