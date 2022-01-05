{%- capture title -%}
ViveknSentimentApproach
{%- endcapture -%}

{%- capture description -%}
Trains a sentiment analyser inspired by the algorithm by Vivek Narayanan https://github.com/vivekn/sentiment/.

The algorithm is based on the paper
["Fast and accurate sentiment classification using an enhanced Naive Bayes model"](https://arxiv.org/abs/1305.6143).

The analyzer requires sentence boundaries to give a score in context.
Tokenization is needed to make sure tokens are within bounds. Transitivity requirements are also required.

The training data needs to consist of a column for normalized text and a label column (either `"positive"` or `"negative"`).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/vivekn-sentiment/VivekNarayanSentimentApproach.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN, DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
SENTIMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

document = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

token = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normal")

vivekn = ViveknSentimentApproach() \
    .setInputCols(["document", "normal"]) \
    .setSentimentCol("train_sentiment") \
    .setOutputCol("result_sentiment")

finisher = Finisher() \
    .setInputCols(["result_sentiment"]) \
    .setOutputCols("final_sentiment")

pipeline = Pipeline().setStages([document, token, normalizer, vivekn, finisher])

training = spark.createDataFrame([
    ("I really liked this movie!", "positive"),
    ("The cast was horrible", "negative"),
    ("Never going to watch this again or recommend it to anyone", "negative"),
    ("It's a waste of time", "negative"),
    ("I loved the protagonist", "positive"),
    ("The music was really really good", "positive")
]).toDF("text", "train_sentiment")
pipelineModel = pipeline.fit(training)

data = spark.createDataFrame([
    ["I recommend this movie"],
    ["Dont waste your time!!!"]
]).toDF("text")
result = pipelineModel.transform(data)

result.select("final_sentiment").show(truncate=False)
+---------------+
|final_sentiment|
+---------------+
|[positive]     |
|[negative]     |
+---------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.Normalizer
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach
import com.johnsnowlabs.nlp.Finisher
import org.apache.spark.ml.Pipeline

val document = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val token = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val normalizer = new Normalizer()
  .setInputCols("token")
  .setOutputCol("normal")

val vivekn = new ViveknSentimentApproach()
  .setInputCols("document", "normal")
  .setSentimentCol("train_sentiment")
  .setOutputCol("result_sentiment")

val finisher = new Finisher()
  .setInputCols("result_sentiment")
  .setOutputCols("final_sentiment")

val pipeline = new Pipeline().setStages(Array(document, token, normalizer, vivekn, finisher))

val training = Seq(
  ("I really liked this movie!", "positive"),
  ("The cast was horrible", "negative"),
  ("Never going to watch this again or recommend it to anyone", "negative"),
  ("It's a waste of time", "negative"),
  ("I loved the protagonist", "positive"),
  ("The music was really really good", "positive")
).toDF("text", "train_sentiment")
val pipelineModel = pipeline.fit(training)

val data = Seq(
  "I recommend this movie",
  "Dont waste your time!!!"
).toDF("text")
val result = pipelineModel.transform(data)

result.select("final_sentiment").show(false)
+---------------+
|final_sentiment|
+---------------+
|[positive]     |
|[negative]     |
+---------------+

{%- endcapture -%}

{%- capture api_link -%}
[ViveknSentimentApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/sda/vivekn/ViveknSentimentApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[ViveknSentimentApproach](https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.ViveknSentimentApproach.html)
{%- endcapture -%}

{%- capture source_link -%}
[ViveknSentimentApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn/ViveknSentimentApproach.scala)
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
