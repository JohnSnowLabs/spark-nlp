{%- capture title -%}
Wav2Vec2ForCTC
{%- endcapture -%}

{%- capture description -%}
Wav2Vec2 Model with a language modeling head on top for Connectionist Temporal Classification
(CTC). Wav2Vec2 was proposed in wav2vec 2.0: A Framework for Self-Supervised Learning of
Speech Representations by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

The annotator takes audio files and transcribes it as text. The audio needs to be provided
pre-processed an array of floats.

Note that this annotator is currently not supported on Apple Silicon processors such as the
M1. This is due to the processor not supporting instructions for XLA.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val speechToText = Wav2Vec2ForCTC.pretrained()
  .setInputCols("audio_assembler")
  .setOutputCol("text")
```
The default model is `"asr_wav2vec2_base_960h"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://nlp.johnsnowlabs.com/models).

To see which models are compatible and how to import them see
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
examples, see
[Wav2Vec2ForCTCTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/audio/Wav2Vec2ForCTCTestSpec.scala).
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
speechToText = Wav2Vec2ForCTC \
    .pretrained() \
    .setInputCols(["audio_assembler"]) \
    .setOutputCol("text")

pipeline = Pipeline().setStages([audioAssembler, speechToText])
processedAudioFloats = spark.createDataFrame([[rawFloats]]).toDF("audio_content")
result = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)
result.select("text.result").show(truncate = False)
+------------------------------------------------------------------------------------------+
|result                                                                                    |
+------------------------------------------------------------------------------------------+
|[MISTER QUILTER IS THE APOSTLE OF THE MIDLE CLASES AND WE ARE GLAD TO WELCOME HIS GOSPEL ]|
+------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.annotators.audio.Wav2Vec2ForCTC
import org.apache.spark.ml.Pipeline

val audioAssembler: AudioAssembler = new AudioAssembler()
  .setInputCol("audio_content")
  .setOutputCol("audio_assembler")

val speechToText: Wav2Vec2ForCTC = Wav2Vec2ForCTC
  .pretrained()
  .setInputCols("audio_assembler")
  .setOutputCol("text")

val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

val bufferedSource =
  scala.io.Source.fromFile("src/test/resources/audio/csv/audi_floats.csv")

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
|[MISTER QUILTER IS THE APOSTLE OF THE MIDLE CLASES AND WE ARE GLAD TO WELCOME HIS GOSPEL ]|
+------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[Wav2Vec2ForCTC](/api/com/johnsnowlabs/nlp/annotators/audio/Wav2Vec2ForCTC)
{%- endcapture -%}

{%- capture python_api_link -%}
[Wav2Vec2ForCTC](/api/python/reference/autosummary/python/sparknlp/annotator/audio/wav2vec2_for_ctc/index.html?highlight=wav2vec2forctc#python.sparknlp.annotator.audio.wav2vec2_for_ctc.Wav2Vec2ForCTC)
{%- endcapture -%}

{%- capture source_link -%}
[Wav2Vec2ForCTC](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/audio/Wav2Vec2ForCTC.scala)
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