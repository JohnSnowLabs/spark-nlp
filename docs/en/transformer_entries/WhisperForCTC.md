{%- capture title -%}
WhisperForCTC
{%- endcapture -%}

{%- capture description -%}
Whisper Model with a language modeling head on top for Connectionist Temporal Classification
(CTC).

Whisper is an automatic speech recognition (ASR) system trained on 680,000 hours of
multilingual and multitask supervised data collected from the web. It transcribe in multiple
languages, as well as translate from those languages into English.

The audio needs to be provided pre-processed an array of floats.

Note that at the moment, this annotator only supports greedy search and only Spark Versions
3.4 and up are supported.

For multilingual models, the language and the task (transcribe or translate) can be set with
`setLanguage` and `setTask`.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val speechToText = WhisperForCTC.pretrained()
  .setInputCols("audio_assembler")
  .setOutputCol("text")
```

The default model is `"asr_whisper_tiny_opt"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models).

To see which models are compatible and how to import them see
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
examples, see
[WhisperForCTCTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/audio/WhisperForCTCTest.scala).

**References:**

[Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)

**Paper Abstract:**

*We study the capabilities of speech processing systems trained simply to predict large
amounts of transcripts of audio on the internet. When scaled to 680,000 hours of multilingual
and multitask supervision, the resulting models generalize well to standard benchmarks and are
often competitive with prior fully supervised results but in a zero- shot transfer setting
without the need for any fine- tuning. When compared to humans, the models approach their
accuracy and robustness. We are releasing models and inference code to serve as a foundation
for further work on robust speech processing.*
{%- endcapture -%}

{%- capture input_anno -%}
AUDIO
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

audioAssembler = AudioAssembler() \
    .setInputCol("audio_content") \
    .setOutputCol("audio_assembler")

speechToText = WhisperForCTC.pretrained() \
    .setInputCols(["audio_assembler"]) \
    .setOutputCol("text")

pipeline = Pipeline().setStages([audioAssembler, speechToText])
processedAudioFloats = spark.createDataFrame([[rawFloats]]).toDF("audio_content")
result = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)
result.select("text.result").show(truncate = False)
+------------------------------------------------------------------------------------------+
|result                                                                                    |
+------------------------------------------------------------------------------------------+
|[ Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.]|
+------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.annotators.audio.WhisperForCTC
import org.apache.spark.ml.Pipeline

val audioAssembler: AudioAssembler = new AudioAssembler()
  .setInputCol("audio_content")
  .setOutputCol("audio_assembler")

val speechToText: WhisperForCTC = WhisperForCTC
  .pretrained()
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
+------------------------------------------------------------------------------------------+
|result                                                                                    |
+------------------------------------------------------------------------------------------+
|[ Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.]|
+------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[WhisperForCTC](/api/com/johnsnowlabs/nlp/annotators/audio/WhisperForCTC)
{%- endcapture -%}

{%- capture python_api_link -%}
[WhisperForCTC](/api/python/reference/autosummary/sparknlp/annotator/audio/whisper_for_ctc/index.html?highlight=whisperforctc#python.sparknlp.annotator.audio.whisper_for_ctc.WhisperForCTC)
{%- endcapture -%}

{%- capture source_link -%}
[WhisperForCTC](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/audio/WhisperForCTC.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}