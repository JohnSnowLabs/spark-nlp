{%- capture title -%}
SentimentDLApproach
{%- endcapture -%}

{%- capture approach_description -%}
Trains a SentimentDL, an annotator for multi-class sentiment analysis.

In natural language processing, sentiment analysis is the task of classifying the affective state or subjective view
of a text. A common example is if either a product review or tweet can be interpreted positively or negatively.

For the instantiated/pretrained models, see SentimentDLModel.

**Notes**:
  - This annotator accepts a label column of a single item in either type of String, Int, Float, or Double.
    So positive sentiment can be expressed as either `"positive"` or `0`, negative sentiment as `"negative"` or `1`.
  - Any type of sentence embeddings, such as the [UniversalSentenceEncoder](/docs/en/transformers#universalsentenceencoder),
    [BertSentenceEmbeddings](/docs/en/transformers#bertsentenceembeddings), or
    [SentenceEmbeddings](/docs/en/annotators#sentenceembeddings) can be used for the `inputCol`.

For extended examples of usage, see the [Examples](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/classification/SentimentDL_train_multiclass_sentiment_classifier.ipynb).
{%- endcapture -%}

{%- capture approach_input_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture approach_python_example -%}
# In this example, `sentiment.csv` is in the form
#
# text,label
# This movie is the best movie I have watched ever! In my opinion this movie can win an award.,0
# This was a terrible movie! The acting was bad really bad!,1
#
# The model can then be trained with

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

smallCorpus = spark.read.option("header", "True").csv("src/test/resources/classifier/sentiment.csv")

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

useEmbeddings = UniversalSentenceEncoder.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")

docClassifier = SentimentDLApproach() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("sentiment") \
    .setLabelColumn("label") \
    .setBatchSize(32) \
    .setMaxEpochs(1) \
    .setLr(5e-3) \
    .setDropout(0.5)

pipeline = Pipeline() \
    .setStages(
      [
        documentAssembler,
        useEmbeddings,
        docClassifier
      ]
    )

pipelineModel = pipeline.fit(smallCorpus)

{%- endcapture -%}

{%- capture approach_scala_example -%}
// In this example, `sentiment.csv` is in the form
//
// text,label
// This movie is the best movie I have watched ever! In my opinion this movie can win an award.,0
// This was a terrible movie! The acting was bad really bad!,1
//
// The model can then be trained with
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.UniversalSentenceEncoder
import com.johnsnowlabs.nlp.annotators.classifier.dl.{SentimentDLApproach, SentimentDLModel}
import org.apache.spark.ml.Pipeline

val smallCorpus = spark.read.option("header", "true").csv("src/test/resources/classifier/sentiment.csv")

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val useEmbeddings = UniversalSentenceEncoder.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence_embeddings")

val docClassifier = new SentimentDLApproach()
  .setInputCols("sentence_embeddings")
  .setOutputCol("sentiment")
  .setLabelColumn("label")
  .setBatchSize(32)
  .setMaxEpochs(1)
  .setLr(5e-3f)
  .setDropout(0.5f)

val pipeline = new Pipeline()
  .setStages(
    Array(
      documentAssembler,
      useEmbeddings,
      docClassifier
    )
  )

val pipelineModel = pipeline.fit(smallCorpus)

{%- endcapture -%}

{%- capture approach_api_link -%}
[SentimentDLApproach](/api/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[SentimentDLApproach](/api/python/reference/autosummary/sparknlp/annotator/sentiment/sentiment_dl/index.html#sparknlp.annotator.sentiment.sentiment_dl.SentimentDLApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[SentimentDLApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLApproach.scala)
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
