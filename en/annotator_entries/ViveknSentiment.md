{%- capture title -%}
ViveknSentiment
{%- endcapture -%}

{%- capture model_description -%}
Sentiment analyser inspired by the algorithm by Vivek Narayanan https://github.com/vivekn/sentiment/.

The algorithm is based on the paper
["Fast and accurate sentiment classification using an enhanced Naive Bayes model"](https://arxiv.org/abs/1305.6143).

This is the instantiated model of the ViveknSentimentApproach.
For training your own model, please see the documentation of that class.

The analyzer requires sentence boundaries to give a score in context.
Tokenization is needed to make sure tokens are within bounds. Transitivity requirements are also required.

For extended examples of usage, see the [Examples](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/sentiment-detection/VivekNarayanSentimentApproach.ipynb)
and the [ViveknSentimentTestSpec](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn).
{%- endcapture -%}

{%- capture model_input_anno -%}
TOKEN, DOCUMENT
{%- endcapture -%}

{%- capture model_output_anno -%}
SENTIMENT
{%- endcapture -%}

{%- capture model_api_link -%}
[ViveknSentimentModel](/api/com/johnsnowlabs/nlp/annotators/sda/vivekn/ViveknSentimentModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[ViveknSentimentModel](/api/python/reference/autosummary/sparknlp/annotator/sentiment/vivekn_sentiment/index.html#sparknlp.annotator.sentiment.vivekn_sentiment.ViveknSentimentModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[ViveknSentimentModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn/ViveknSentimentModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Trains a sentiment analyser inspired by the algorithm by Vivek Narayanan https://github.com/vivekn/sentiment/.

The algorithm is based on the paper
["Fast and accurate sentiment classification using an enhanced Naive Bayes model"](https://arxiv.org/abs/1305.6143).

The analyzer requires sentence boundaries to give a score in context.
Tokenization is needed to make sure tokens are within bounds. Transitivity requirements are also required.

The training data needs to consist of a column for normalized text and a label column (either `"positive"` or `"negative"`).

For extended examples of usage, see the [Examples](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/sentiment-detection/VivekNarayanSentimentApproach.ipynb)
and the [ViveknSentimentTestSpec](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn).
{%- endcapture -%}

{%- capture approach_input_anno -%}
TOKEN, DOCUMENT
{%- endcapture -%}

{%- capture approach_output_anno -%}
SENTIMENT
{%- endcapture -%}

{%- capture approach_python_example -%}
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

{%- capture approach_scala_example -%}
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

{%- capture approach_api_link -%}
[ViveknSentimentApproach](/api/com/johnsnowlabs/nlp/annotators/sda/vivekn/ViveknSentimentApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[ViveknSentimentApproach](/api/python/reference/autosummary/sparknlp/annotator/sentiment/vivekn_sentiment/index.html#sparknlp.annotator.sentiment.vivekn_sentiment.ViveknSentimentApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[ViveknSentimentApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn/ViveknSentimentApproach.scala)
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
