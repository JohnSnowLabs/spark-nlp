{%- capture title -%}
WordSegmenterApproach
{%- endcapture -%}

{%- capture description -%}
Trains a WordSegmenter which tokenizes non-english or non-whitespace separated texts.

Many languages are not whitespace separated and their sentences are a concatenation of many symbols, like Korean,
Japanese or Chinese. Without understanding the language, splitting the words into their corresponding tokens is
impossible. The WordSegmenter is trained to understand these languages and split them into semantically correct parts.

To train your own model, a training dataset consisting of
[Part-Of-Speech tags](https://en.wikipedia.org/wiki/Part-of-speech_tagging) is required. The data has to be loaded
into a dataframe, where the column is an [Annotation](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/Annotation) of type `"POS"`. This can be
set with `setPosColumn`.

**Tip**: The helper class [POS](#pos-dataset) might be useful to read training data into data frames.

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter/annotation/chinese/word_segmentation).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture python_example -%}
# In this example, `"chinese_train.utf8"` is in the form of
#
# 十|LL 四|RR 不|LL 是|RR 四|LL 十|RR
#
# and is loaded with the `POS` class to create a dataframe of `"POS"` type Annotations.

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

wordSegmenter = WordSegmenterApproach() \
    .setInputCols(["document"]) \
    .setOutputCol("token") \
    .setPosColumn("tags") \
    .setNIterations(5)

pipeline = Pipeline().setStages([
    documentAssembler,
    wordSegmenter
])

trainingDataSet = POS().readDataset(
    spark,
    "src/test/resources/word-segmenter/chinese_train.utf8"
)

pipelineModel = pipeline.fit(trainingDataSet)

{%- endcapture -%}

{%- capture scala_example -%}
// In this example, `"chinese_train.utf8"` is in the form of
//
// 十|LL 四|RR 不|LL 是|RR 四|LL 十|RR
//
// and is loaded with the `POS` class to create a dataframe of `"POS"` type Annotations.

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.ws.WordSegmenterApproach
import com.johnsnowlabs.nlp.training.POS
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val wordSegmenter = new WordSegmenterApproach()
  .setInputCols("document")
  .setOutputCol("token")
  .setPosColumn("tags")
  .setNIterations(5)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  wordSegmenter
))

val trainingDataSet = POS().readDataset(
  ResourceHelper.spark,
  "src/test/resources/word-segmenter/chinese_train.utf8"
)

val pipelineModel = pipeline.fit(trainingDataSet)

{%- endcapture -%}

{%- capture api_link -%}
[WordSegmenterApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ws/WordSegmenterApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[WordSegmenterApproach](/api/python/reference/autosummary/python/sparknlp/annotator/ws/word_segmenter/index.html#sparknlp.annotator.ws.word_segmenter.WordSegmenterApproach)
{%- endcapture -%}

{%- capture source_link -%}
[WordSegmenterApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ws/WordSegmenterApproach.scala)
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

