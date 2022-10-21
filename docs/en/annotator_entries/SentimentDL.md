{%- capture title -%}
SentimentDL
{%- endcapture -%}

{%- capture model_description -%}
SentimentDL, an annotator for multi-class sentiment analysis.

In natural language processing, sentiment analysis is the task of classifying the affective state or subjective view
of a text. A common example is if either a product review or tweet can be interpreted positively or negatively.

This is the instantiated model of the SentimentDLApproach.
For training your own model, please see the documentation of that class.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val sentiment = SentimentDLModel.pretrained()
  .setInputCols("sentence_embeddings")
  .setOutputCol("sentiment")
```
The default model is `"sentimentdl_use_imdb"`, if no name is provided. It is english sentiment analysis trained on
the IMDB dataset.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Sentiment+Analysis).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb)
and the [SentimentDLTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLTestSpec.scala).
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
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")

sentiment = SentimentDLModel.pretrained("sentimentdl_use_twitter") \
    .setInputCols(["sentence_embeddings"]) \
    .setThreshold(0.7) \
    .setOutputCol("sentiment")

pipeline = Pipeline().setStages([
    documentAssembler,
    useEmbeddings,
    sentiment
])

data = spark.createDataFrame([
    ["Wow, the new video is awesome!"],
    ["bruh what a damn waste of time"]
]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("text", "sentiment.result").show(truncate=False)
+------------------------------+----------+
|text                          |result    |
+------------------------------+----------+
|Wow, the new video is awesome!|[positive]|
|bruh what a damn waste of time|[negative]|
+------------------------------+----------+

{%- endcapture -%}

{%- capture model_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.UniversalSentenceEncoder
import com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLModel
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val useEmbeddings = UniversalSentenceEncoder.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence_embeddings")

val sentiment = SentimentDLModel.pretrained("sentimentdl_use_twitter")
  .setInputCols("sentence_embeddings")
  .setThreshold(0.7F)
  .setOutputCol("sentiment")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  useEmbeddings,
  sentiment
))

val data = Seq(
  "Wow, the new video is awesome!",
  "bruh what a damn waste of time"
).toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("text", "sentiment.result").show(false)
+------------------------------+----------+
|text                          |result    |
+------------------------------+----------+
|Wow, the new video is awesome!|[positive]|
|bruh what a damn waste of time|[negative]|
+------------------------------+----------+

{%- endcapture -%}

{%- capture model_api_link -%}
[SentimentDLModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[SentimentDLModel](/api/python/reference/autosummary/python/sparknlp/annotator/sentiment/sentiment_dl/index.html#sparknlp.annotator.sentiment.sentiment_dl.SentimentDLModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[SentimentDLModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Trains a SentimentDL, an annotator for multi-class sentiment analysis.

In natural language processing, sentiment analysis is the task of classifying the affective state or subjective view
of a text. A common example is if either a product review or tweet can be interpreted positively or negatively.

For the instantiated/pretrained models, see SentimentDLModel.

**Notes**:
  - This annotator accepts a label column of a single item in either type of
    String, Int, Float, or Double. So positive sentiment can be expressed as
    either `"positive"` or `0`, negative sentiment as `"negative"` or `1`.
  - [UniversalSentenceEncoder](/docs/en/transformers#universalsentenceencoder),
    [BertSentenceEmbeddings](/docs/en/transformers#bertsentenceembeddings),
    [SentenceEmbeddings](/docs/en/annotators#sentenceembeddings) or other
    sentence based embeddings can be used

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/SentimentDL_train_multiclass_sentiment_classifier.ipynb)
and the [SentimentDLTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLTestSpec.scala).
{%- endcapture -%}

{%- capture approach_input_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
# In this example, `sentiment.csv` is in the form
#
# text,label
# This movie is the best movie I have watched ever! In my opinion this movie can win an award.,0
# This was a terrible movie! The acting was bad really bad!,1
#
# The model can then be trained with

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
[SentimentDLApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[SentimentDLApproach](/api/python/reference/autosummary/python/sparknlp/annotator/sentiment/sentiment_dl/index.html#sparknlp.annotator.sentiment.sentiment_dl.SentimentDLApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[SentimentDLApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLApproach.scala)
{%- endcapture -%}


{% include templates/approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
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
%}
