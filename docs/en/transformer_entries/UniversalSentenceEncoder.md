{%- capture title -%}
UniversalSentenceEncoder
{%- endcapture -%}

{%- capture description -%}
The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val useEmbeddings = UniversalSentenceEncoder.pretrained()
  .setInputCols("sentence")
  .setOutputCol("sentence_embeddings")
```
The default model is `"tfhub_use"`, if no name is provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Embeddings).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb)
and the [UniversalSentenceEncoderTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/UniversalSentenceEncoderTestSpec.scala).

**Sources:**

[Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)

https://tfhub.dev/google/universal-sentence-encoder/2

**Paper abstract:**

*We present models for encoding sentences into embedding vectors that specifically target transfer learning to other
NLP tasks. The models are efficient and result in accurate performance on diverse transfer tasks. Two variants of the
encoding models allow for trade-offs between accuracy and compute resources. For both variants, we investigate and
report the relationship between model complexity, resource consumption, the availability of transfer task training
data, and task performance. Comparisons are made with baselines that use word level transfer learning via pretrained
word embeddings as well as baselines do not use any transfer learning. We find that transfer learning using sentence
embeddings tends to outperform word level transfer. With transfer learning via sentence embeddings, we observe
surprisingly good performance with minimal amounts of supervised training data for a transfer task. We obtain
encouraging results on Word Embedding Association Tests (WEAT) targeted at detecting model bias. Our pre-trained
sentence encoding models are made freely available for download and on TF Hub.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture api_link -%}
[UniversalSentenceEncoder](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/UniversalSentenceEncoder)
{%- endcapture -%}

{%- capture python_api_link -%}
[UniversalSentenceEncoder](/api/python/reference/autosummary/python/sparknlp/annotator/embeddings/universal_sentence_encoder/index.html#sparknlp.annotator.embeddings.universal_sentence_encoder.UniversalSentenceEncoder)
{%- endcapture -%}

{%- capture source_link -%}
[UniversalSentenceEncoder](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/UniversalSentenceEncoder.scala)
{%- endcapture -%}

{%- capture prediction_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

# Use the transformer embeddings
embeddings = UniversalSentenceEncoder.pretrained('tfhub_use', lang="en") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")

# This pretrained model requires those specific transformer embeddings
classifier = SentimentDLModel().pretrained('sentimentdl_use_imdb') \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("sentiment")

pipeline = Pipeline().setStages([
    documentAssembler,
    embeddings,
    classifier
])

data = spark.createDataFrame([["That was a fantastic movie!"]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("sentiment.result").show(truncate=False)
+------+
|result|
+------+
|[pos] |
+------+
{%- endcapture -%}

{%- capture prediction_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
import com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLModel
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = UniversalSentenceEncoder.pretrained("tfhub_use", lang = "en")
  .setInputCols("document")
  .setOutputCol("sentence_embeddings")

// This pretrained model requires those specific transformer embeddings
val classifier = SentimentDLModel.pretrained("sentimentdl_use_imdb")
  .setInputCols("sentence_embeddings")
  .setOutputCol("sentiment")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  embeddings,
  classifier
))

val data = Seq("That was a fantastic movie!").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("sentiment.result").show(false)
+------+
|result|
+------+
|[pos] |
{%- endcapture -%}

{%- capture training_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

# Use the transformer embeddings
embeddings = UniversalSentenceEncoder.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")

# Then the training can start with the transformer embeddings
classifier = SentimentDLApproach() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("sentiment") \
    .setLabelColumn("label") \
    .setBatchSize(32) \
    .setMaxEpochs(1) \
    .setLr(5e-3) \
    .setDropout(0.5)

pipeline = Pipeline().setStages([
    documentAssembler,
    embeddings,
    classifier
])

smallCorpus = spark.read.option("header", "True").csv("sentiment.csv")
result = pipeline.fit(smallCorpus)
{%- endcapture -%}

{%- capture training_scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.UniversalSentenceEncoder
import com.johnsnowlabs.nlp.annotators.classifier.dl.{SentimentDLApproach, SentimentDLModel}
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

// Use the transformer embeddings
val useEmbeddings = UniversalSentenceEncoder.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence_embeddings")

// Then the training can start with the transformer embeddings
val docClassifier = new SentimentDLApproach()
  .setInputCols("sentence_embeddings")
  .setOutputCol("sentiment")
  .setLabelColumn("label")
  .setBatchSize(32)
  .setMaxEpochs(1)
  .setLr(5e-3f)
  .setDropout(0.5f)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  useEmbeddings,
  docClassifier
))

val smallCorpus = spark.read.option("header", "true").csv("sentiment.csv")
val pipelineModel = pipeline.fit(smallCorpus)
{%- endcapture -%}

{%- capture embeddings_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

embeddings = UniversalSentenceEncoder.pretrained() \
    .setInputCols(["sentence"]) \
    .setOutputCol("sentence_embeddings")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      sentence,
      embeddings,
      embeddingsFinisher
    ])

data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[0.04616805538535118,0.022307956591248512,-0.044395286589860916,-0.0016493503...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture embeddings_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val embeddings = UniversalSentenceEncoder.pretrained()
  .setInputCols("sentence")
  .setOutputCol("sentence_embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)
  .setCleanAnnotations(false)

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    sentence,
    embeddings,
    embeddingsFinisher
  ))

val data = Seq("This is a sentence.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[0.04616805538535118,0.022307956591248512,-0.044395286589860916,-0.0016493503...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{% include templates/transformer_usecases_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_api_link=python_api_link
api_link=api_link
source_link=source_link
prediction_python_example=prediction_python_example
prediction_scala_example=prediction_scala_example
training_python_example=training_python_example
training_scala_example=training_scala_example
embeddings_python_example=embeddings_python_example
embeddings_scala_example=embeddings_scala_example
%}