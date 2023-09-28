{%- capture title -%}
SentimentDetector
{%- endcapture -%}

{%- capture model_description -%}
Rule based sentiment detector, which calculates a score based on predefined keywords.

This is the instantiated model of the SentimentDetector.
For training your own model, please see the documentation of that class.

A dictionary of predefined sentiment keywords must be provided with `setDictionary`, where each line is a word
delimited to its class (either `positive` or `negative`).
The dictionary can be set as a delimited text file.

By default, the sentiment score will be assigned labels `"positive"` if the score is `>= 0`, else `"negative"`.
To retrieve the raw sentiment scores, `enableScore` needs to be set to `true`.

For extended examples of usage, see the [Examples](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/sentiment-detection/RuleBasedSentiment.ipynb)
and the [SentimentTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sda/pragmatic/PragmaticSentimentTestSpec.scala).
{%- endcapture -%}

{%- capture model_input_anno -%}
TOKEN, DOCUMENT
{%- endcapture -%}

{%- capture model_output_anno -%}
SENTIMENT
{%- endcapture -%}

{%- capture model_api_link -%}
[SentimentDetectorModel](/api/com/johnsnowlabs/nlp/annotators/sda/pragmatic/SentimentDetectorModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[SentimentDetectorModel](/api/python/reference/autosummary/sparknlp/annotator/sentiment/sentiment_detector/index.html#sparknlp.annotator.sentiment.sentiment_detector.SentimentDetectorModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[SentimentDetectorModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/pragmatic/SentimentDetectorModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Trains a rule based sentiment detector, which calculates a score based on predefined keywords.

A dictionary of predefined sentiment keywords must be provided with `setDictionary`, where each line is a word
delimited to its class (either `positive` or `negative`).
The dictionary can be set as a delimited text file.

By default, the sentiment score will be assigned labels `"positive"` if the score is `>= 0`, else `"negative"`.
To retrieve the raw sentiment scores, `enableScore` needs to be set to `true`.

For extended examples of usage, see the [Examples](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/sentiment-detection/RuleBasedSentiment.ipynb)
and the [SentimentTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sda/pragmatic/PragmaticSentimentTestSpec.scala).
{%- endcapture -%}

{%- capture approach_input_anno -%}
TOKEN, DOCUMENT
{%- endcapture -%}

{%- capture approach_output_anno -%}
SENTIMENT
{%- endcapture -%}

{%- capture approach_python_example -%}
# In this example, the dictionary `default-sentiment-dict.txt` has the form of
#
# ...
# cool,positive
# superb,positive
# bad,negative
# uninspired,negative
# ...
#
# where each sentiment keyword is delimited by `","`.

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

lemmatizer = Lemmatizer() \
    .setInputCols(["token"]) \
    .setOutputCol("lemma") \
    .setDictionary("lemmas_small.txt", "->", "\t")

sentimentDetector = SentimentDetector() \
    .setInputCols(["lemma", "document"]) \
    .setOutputCol("sentimentScore") \
    .setDictionary("default-sentiment-dict.txt", ",", ReadAs.TEXT)

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    lemmatizer,
    sentimentDetector,
])

data = spark.createDataFrame([
    ["The staff of the restaurant is nice"],
    ["I recommend others to avoid because it is too expensive"]
]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("sentimentScore.result").show(truncate=False)
+----------+  #  +------+ for enableScore set to True
|result    |  #  |result|
+----------+  #  +------+
|[positive]|  #  |[1.0] |
|[negative]|  #  |[-2.0]|
+----------+  #  +------+

{%- endcapture -%}

{%- capture approach_scala_example -%}
// In this example, the dictionary `default-sentiment-dict.txt` has the form of
//
// ...
// cool,positive
// superb,positive
// bad,negative
// uninspired,negative
// ...
//
// where each sentiment keyword is delimited by `","`.

import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.annotators.Lemmatizer
import com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector
import com.johnsnowlabs.nlp.util.io.ReadAs
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val lemmatizer = new Lemmatizer()
  .setInputCols("token")
  .setOutputCol("lemma")
  .setDictionary("src/test/resources/lemma-corpus-small/lemmas_small.txt", "->", "\t")

val sentimentDetector = new SentimentDetector()
  .setInputCols("lemma", "document")
  .setOutputCol("sentimentScore")
  .setDictionary("src/test/resources/sentiment-corpus/default-sentiment-dict.txt", ",", ReadAs.TEXT)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  lemmatizer,
  sentimentDetector,
))

val data = Seq(
  "The staff of the restaurant is nice",
  "I recommend others to avoid because it is too expensive"
).toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("sentimentScore.result").show(false)
+----------+  //  +------+ for enableScore set to true
|result    |  //  |result|
+----------+  //  +------+
|[positive]|  //  |[1.0] |
|[negative]|  //  |[-2.0]|
+----------+  //  +------+

{%- endcapture -%}

{%- capture approach_api_link -%}
[SentimentDetector](/api/com/johnsnowlabs/nlp/annotators/sda/pragmatic/SentimentDetector)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[SentimentDetector](/api/python/reference/autosummary/sparknlp/annotator/sentiment/sentiment_detector/index.html#sparknlp.annotator.sentiment.sentiment_detector.SentimentDetector)
{%- endcapture -%}

{%- capture approach_source_link -%}
[SentimentDetector](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sda/pragmatic/SentimentDetector.scala)
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
