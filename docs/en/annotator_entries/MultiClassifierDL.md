{%- capture title -%}
MultiClassifierDL
{%- endcapture -%}

{%- capture model_description -%}
MultiClassifierDL for Multi-label Text Classification.

MultiClassifierDL Bidirectional GRU with Convolution model we have built inside TensorFlow and supports up to 100 classes.
The input to MultiClassifierDL are Sentence Embeddings such as state-of-the-art
[UniversalSentenceEncoder](/docs/en/transformers#universalsentenceencoder),
[BertSentenceEmbeddings](/docs/en/transformers#bertsentenceembeddings) or
[SentenceEmbeddings](/docs/en/annotators#sentenceembeddings).

This is the instantiated model of the MultiClassifierDLApproach.
For training your own model, please see the documentation of that class.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val multiClassifier = MultiClassifierDLModel.pretrained()
  .setInputCols("sentence_embeddings")
  .setOutputCol("categories")
```
The default model is `"multiclassifierdl_use_toxic"`, if no name is provided. It uses embeddings from the
[UniversalSentenceEncoder](/docs/en/transformers#universalsentenceencoder) and classifies toxic comments.
The data is based on the
[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview).
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Text+Classification).

In machine learning, multi-label classification and the strongly related problem of multi-output classification are
variants of the classification problem where multiple labels may be assigned to each instance. Multi-label
classification is a generalization of multiclass classification, which is the single-label problem of categorizing
instances into precisely one of more than two classes; in the multi-label problem there is no constraint on how many
of the classes the instance can be assigned to.
Formally, multi-label classification is the problem of finding a model that maps inputs x to binary vectors y
(assigning a value of 0 or 1 for each element (label) in y).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/MultiClassifierDL_train_multi_label_E2E_challenge_classifier.ipynb)
and the [MultiClassifierDLTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLTestSpec.scala).
{%- endcapture -%}

{%- capture model_input_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture model_output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture model_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

useEmbeddings = UniversalSentenceEncoder.pretrained() \
    .setInputCols("document") \
    .setOutputCol("sentence_embeddings")

multiClassifierDl = MultiClassifierDLModel.pretrained() \
    .setInputCols("sentence_embeddings") \
    .setOutputCol("classifications")

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      useEmbeddings,
      multiClassifierDl
    ])

data = spark.createDataFrame([
    ["This is pretty good stuff!"],
    ["Wtf kind of crap is this"]
]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("text", "classifications.result").show(truncate=False)
+--------------------------+----------------+
|text                      |result          |
+--------------------------+----------------+
|This is pretty good stuff!|[]              |
|Wtf kind of crap is this  |[toxic, obscene]|
+--------------------------+----------------+

{%- endcapture -%}

{%- capture model_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLModel
import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val useEmbeddings = UniversalSentenceEncoder.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence_embeddings")

val multiClassifierDl = MultiClassifierDLModel.pretrained()
  .setInputCols("sentence_embeddings")
  .setOutputCol("classifications")

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    useEmbeddings,
    multiClassifierDl
  ))

val data = Seq(
  "This is pretty good stuff!",
  "Wtf kind of crap is this"
).toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("text", "classifications.result").show(false)
+--------------------------+----------------+
|text                      |result          |
+--------------------------+----------------+
|This is pretty good stuff!|[]              |
|Wtf kind of crap is this  |[toxic, obscene]|
+--------------------------+----------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[MultiClassifierDLModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[MultiClassifierDLModel](/api/python/reference/autosummary/python/sparknlp/annotator/classifier_dl/multi_classifier_dl/index.html#sparknlp.annotator.classifier_dl.multi_classifier_dl.MultiClassifierDLModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[MultiClassifierDLModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Trains a MultiClassifierDL for Multi-label Text Classification.

MultiClassifierDL uses a Bidirectional GRU with a convolutional model that we have built inside TensorFlow and supports
up to 100 classes.

For instantiated/pretrained models, see MultiClassifierDLModel.

The input to `MultiClassifierDL` are Sentence Embeddings such as the state-of-the-art
[UniversalSentenceEncoder](/docs/en/transformers#universalsentenceencoder),
[BertSentenceEmbeddings](/docs/en/transformers#bertsentenceembeddings) or
[SentenceEmbeddings](/docs/en/annotators#sentenceembeddings).

In machine learning, multi-label classification and the strongly related problem of multi-output classification are
variants of the classification problem where multiple labels may be assigned to each instance. Multi-label
classification is a generalization of multiclass classification, which is the single-label problem of categorizing
instances into precisely one of more than two classes; in the multi-label problem there is no constraint on how many
of the classes the instance can be assigned to.
Formally, multi-label classification is the problem of finding a model that maps inputs x to binary vectors y
(assigning a value of 0 or 1 for each element (label) in y).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/MultiClassifierDL_train_multi_label_E2E_challenge_classifier.ipynb)
and the [MultiClassifierDLTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLTestSpec.scala).
{%- endcapture -%}

{%- capture approach_input_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture approach_python_example -%}
# In this example, the training data has the form
#
# +----------------+--------------------+--------------------+
# |              id|                text|              labels|
# +----------------+--------------------+--------------------+
# |ed58abb40640f983|PN NewsYou mean ... |             [toxic]|
# |a1237f726b5f5d89|Dude.  Place the ...|   [obscene, insult]|
# |24b0d6c8733c2abe|Thanks  - thanks ...|            [insult]|
# |8c4478fb239bcfc0|" Gee, 5 minutes ...|[toxic, obscene, ...|
# +----------------+--------------------+--------------------+

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# Process training data to create text with associated array of labels

trainDataset.printSchema()
# root
#  |-- id: string (nullable = true)
#  |-- text: string (nullable = true)
#  |-- labels: array (nullable = true)
#  |    |-- element: string (containsNull = true)


# Then create pipeline for training
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document") \
    .setCleanupMode("shrink")

embeddings = UniversalSentenceEncoder.pretrained() \
    .setInputCols("document") \
    .setOutputCol("embeddings")

docClassifier = MultiClassifierDLApproach() \
    .setInputCols("embeddings") \
    .setOutputCol("category") \
    .setLabelColumn("labels") \
    .setBatchSize(128) \
    .setMaxEpochs(10) \
    .setLr(1e-3) \
    .setThreshold(0.5) \
    .setValidationSplit(0.1)

pipeline = Pipeline() \
    .setStages(
      [
        documentAssembler,
        embeddings,
        docClassifier
      ]
    )

pipelineModel = pipeline.fit(trainDataset)

{%- endcapture -%}

{%- capture approach_scala_example -%}
// In this example, the training data has the form (Note: labels can be arbitrary)
//
// mr,ref
// "name[Alimentum], area[city centre], familyFriendly[no], near[Burger King]",Alimentum is an adult establish found in the city centre area near Burger King.
// "name[Alimentum], area[city centre], familyFriendly[yes]",Alimentum is a family-friendly place in the city centre.
// ...
//
// It needs some pre-processing first, so the labels are of type `Array[String]`. This can be done like so:

import spark.implicits._
import com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{col, udf}

// Process training data to create text with associated array of labels
def splitAndTrim = udf { labels: String =>
  labels.split(", ").map(x=>x.trim)
}

val smallCorpus = spark.read
  .option("header", true)
  .option("inferSchema", true)
  .option("mode", "DROPMALFORMED")
  .csv("src/test/resources/classifier/e2e.csv")
  .withColumn("labels", splitAndTrim(col("mr")))
  .withColumn("text", col("ref"))
  .drop("mr")

smallCorpus.printSchema()
// root
// |-- ref: string (nullable = true)
// |-- labels: array (nullable = true)
// |    |-- element: string (containsNull = true)

// Then create pipeline for training
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")
  .setCleanupMode("shrink")

val embeddings = UniversalSentenceEncoder.pretrained()
  .setInputCols("document")
  .setOutputCol("embeddings")

val docClassifier = new MultiClassifierDLApproach()
  .setInputCols("embeddings")
  .setOutputCol("category")
  .setLabelColumn("labels")
  .setBatchSize(128)
  .setMaxEpochs(10)
  .setLr(1e-3f)
  .setThreshold(0.5f)
  .setValidationSplit(0.1f)

val pipeline = new Pipeline()
  .setStages(
    Array(
      documentAssembler,
      embeddings,
      docClassifier
    )
  )

val pipelineModel = pipeline.fit(smallCorpus)

{%- endcapture -%}

{%- capture approach_api_link -%}
[MultiClassifierDLApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[MultiClassifierDLApproach](/api/python/reference/autosummary/python/sparknlp/annotator/classifier_dl/multi_classifier_dl/index.html#sparknlp.annotator.classifier_dl.multi_classifier_dl.MultiClassifierDLApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[MultiClassifierDLApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLApproach.scala)
{%- endcapture -%}


{% include templates/approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_example=model_python_example
model_scala_example=model_scala_example
model_python_api_link=model_python_api_link
model_api_link=model_api_link
model_source_link=model_source_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_python_api_link=approach_python_api_link
approach_api_link=approach_api_link
approach_source_link=approach_source_link
approach_note="This annotator accepts a label column of a single item in either type of String, Int, Float, or Double. UniversalSentenceEncoder, BertSentenceEmbeddings, SentenceEmbeddings or other sentence based embeddings can be used for the inputCol"
%}
