{%- capture title -%}
SentenceDetectorDLApproach
{%- endcapture -%}

{%- capture description -%}
Trains an annotator that detects sentence boundaries using a deep learning approach.

For pretrained models see SentenceDetectorDLModel.

Currently, only the CNN model is supported for training, but in the future the architecture of the model can
be set with `setModelArchitecture`.

The default model `"cnn"` is based on the paper
[Deep-EOS: General-Purpose Neural Networks for Sentence Boundary Detection (2020, Stefan Schweter, Sajawel Ahmed)](https://konvens.org/proceedings/2019/papers/KONVENS2019_paper_41.pdf)
using a CNN architecture. We also modified the original implementation a little bit to cover broken sentences and some impossible end of line chars.

Each extracted sentence can be returned in an Array or exploded to separate rows,
if `explodeSentences` is set to `true`.

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/9.SentenceDetectorDL.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
# The training process needs data, where each data point is a sentence.
# In this example the `train.txt` file has the form of
#
# ...
# Slightly more moderate language would make our present situation – namely the lack of progress – a little easier.
# His political successors now have great responsibilities to history and to the heritage of values bequeathed to them by Nelson Mandela.
# ...
#
# where each line is one sentence.
# Training can then be started like so:

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

trainingData = spark.read.text("train.txt").toDF("text")

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLApproach() \
    .setInputCols(["document"]) \
    .setOutputCol("sentences") \
    .setEpochsNumber(100)

pipeline = Pipeline().setStages([documentAssembler, sentenceDetector])

model = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture scala_example -%}
// The training process needs data, where each data point is a sentence.
// In this example the `train.txt` file has the form of
//
// ...
// Slightly more moderate language would make our present situation – namely the lack of progress – a little easier.
// His political successors now have great responsibilities to history and to the heritage of values bequeathed to them by Nelson Mandela.
// ...
//
// where each line is one sentence.
// Training can then be started like so:

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLApproach
import org.apache.spark.ml.Pipeline

val trainingData = spark.read.text("train.txt").toDF("text")

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetectorDLApproach()
  .setInputCols(Array("document"))
  .setOutputCol("sentences")
  .setEpochsNumber(100)

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector))

val model = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture api_link -%}
[SentenceDetectorDLApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/sentence_detector_dl/SentenceDetectorDLApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[SentenceDetectorDLApproach](/api/python/reference/autosummary/python/sparknlp/annotator/sentence/sentence_detector_dl/index.html#sparknlp.annotator.sentence.sentence_detector_dl.SentenceDetectorDLApproach)
{%- endcapture -%}

{%- capture source_link -%}
[SentenceDetectorDLApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sentence_detector_dl/SentenceDetectorDLApproach.scala)
{%- endcapture -%}


{% include templates/training_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
source_link=source_link
%}
