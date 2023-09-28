{%- capture title -%}
HubertForCTC
{%- endcapture -%}

{%- capture description -%}
Hubert Model with a language modeling head on top for Connectionist Temporal Classification
(CTC). Hubert was proposed in HuBERT: Self-Supervised Speech Representation Learning by Masked
Prediction of Hidden Units by Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal
Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed.

The annotator takes audio files and transcribes it as text. The audio needs to be provided
pre-processed an array of floats.

Note that this annotator is currently not supported on Apple Silicon processors such as the
M1. This is due to the processor not supporting instructions for XLA.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val speechToText = HubertForCTC.pretrained()
  .setInputCols("audio_assembler")
  .setOutputCol("text")
```
The default model is `"asr_hubert_large_ls960"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models).

To see which models are compatible and how to import them see
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
examples, see
[HubertForCTCTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/audio/HubertForCTCTest.scala).

**References:**

[HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447)

**Paper Abstract:**

*Self-supervised approaches for speech representation learning are challenged by three unique
problems: (1) there are multiple sound units in each input utterance, (2) there is no lexicon
of input sound units during the pre-training phase, and (3) sound units have variable lengths
with no explicit segmentation. To deal with these three problems, we propose the Hidden-Unit
BERT (HuBERT) approach for self-supervised speech representation learning, which utilizes an
offline clustering step to provide aligned target labels for a BERT-like prediction loss. A
key ingredient of our approach is applying the prediction loss over the masked regions only,
which forces the model to learn a combined acoustic and language model over the continuous
inputs. HuBERT relies primarily on the consistency of the unsupervised clustering step rather
than the intrinsic quality of the assigned cluster labels. Starting with a simple k-means
teacher of 100 clusters, and using two iterations of clustering, the HuBERT model either
matches or improves upon the state-of-the-art wav2vec 2.0 performance on the Librispeech
(960h) and Libri-light (60,000h) benchmarks with 10min, 1h, 10h, 100h, and 960h fine-tuning
subsets. Using a 1B parameter model, HuBERT shows up to 19% and 13% relative WER reduction on
the more challenging dev-other and test-other evaluation subsets.*
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

audioAssembler = AudioAssembler() \\
    .setInputCol("audio_content") \\
    .setOutputCol("audio_assembler")

speechToText = HubertForCTC \\
    .pretrained() \\
    .setInputCols(["audio_assembler"]) \\
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
import com.johnsnowlabs.nlp.annotators.audio.HubertForCTC
import org.apache.spark.ml.Pipeline

val audioAssembler: AudioAssembler = new AudioAssembler()
  .setInputCol("audio_content")
  .setOutputCol("audio_assembler")

val speechToText: HubertForCTC = HubertForCTC
  .pretrained()
  .setInputCols("audio_assembler")
  .setOutputCol("text")

val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

val bufferedSource =
  scala.io.Source.fromFile("src/test/resources/audio/csv/audio_floats.csv")

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
[HubertForCTC](/api/com/johnsnowlabs/nlp/annotators/audio/HubertForCTC)
{%- endcapture -%}

{%- capture python_api_link -%}
[HubertForCTC](/api/python/reference/autosummary/sparknlp/annotator/audio/hubert_for_ctc/index.html#python.sparknlp.annotator.audio.hubert_for_ctc.HubertForCTC)
{%- endcapture -%}

{%- capture source_link -%}
[HubertForCTC](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/audio/HubertForCTC.scala)
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