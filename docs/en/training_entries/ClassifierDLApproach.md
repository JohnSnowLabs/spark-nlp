{%- capture title -%}
ClassifierDLApproach
{%- endcapture -%}

{%- capture description -%}
Trains a ClassifierDL for generic Multi-class Text Classification.

ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an input for text classifications.
The ClassifierDL annotator uses a deep learning model (DNNs) we have built inside TensorFlow and supports up to
100 classes.

For extended examples of usage, see the Spark NLP Workshop
[[1] ](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/scala/training/Train%20Multi-Class%20Text%20Classification%20on%20News%20Articles.scala)
[[2] ](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture python_example -%}
# In this example, the training data `"sentiment.csv"` has the form of
#
# text,label
# This movie is the best movie I have wached ever! In my opinion this movie can win an award.,0
# This was a terrible movie! The acting was bad really bad!,1
# ...
#
# Then traning can be done like so:

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

smallCorpus = spark.read.option("header","True").csv("src/test/resources/classifier/sentiment.csv")

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

useEmbeddings = UniversalSentenceEncoder.pretrained() \
    .setInputCols("document") \
    .setOutputCol("sentence_embeddings")

docClassifier = ClassifierDLApproach() \
    .setInputCols("sentence_embeddings") \
    .setOutputCol("category") \
    .setLabelColumn("label") \
    .setBatchSize(64) \
    .setMaxEpochs(20) \
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

{%- capture scala_example -%}
// In this example, the training data `"sentiment.csv"` has the form of
//
// text,label
// This movie is the best movie I have wached ever! In my opinion this movie can win an award.,0
// This was a terrible movie! The acting was bad really bad!,1
// ...
//
// Then traning can be done like so:

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach
import org.apache.spark.ml.Pipeline

val smallCorpus = spark.read.option("header","true").csv("src/test/resources/classifier/sentiment.csv")

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val useEmbeddings = UniversalSentenceEncoder.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence_embeddings")

val docClassifier = new ClassifierDLApproach()
  .setInputCols("sentence_embeddings")
  .setOutputCol("category")
  .setLabelColumn("label")
  .setBatchSize(64)
  .setMaxEpochs(20)
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

{%- capture api_link -%}
[ClassifierDLApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[ClassifierDLApproach](/api/python/reference/autosummary/python/sparknlp/annotator/classifier_dl/classifier_dl/index.html#sparknlp.annotator.classifier_dl.classifier_dl.ClassifierDLApproach)
{%- endcapture -%}

{%- capture source_link -%}
[ClassifierDLApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLApproach.scala)
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
note="This annotator accepts a label column of a single item in either type of String, Int, Float, or Double. UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings can be used for the inputCol"
%}
