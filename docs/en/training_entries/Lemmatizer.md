{%- capture title -%}
Lemmatizer
{%- endcapture -%}

{%- capture description -%}
Class to find lemmas out of words with the objective of returning a base dictionary word.
Retrieves the significant part of a word. A dictionary of predefined lemmas must be provided with `setDictionary`.
The dictionary can be set as a delimited text file.
Pretrained models can be loaded with `LemmatizerModel.pretrained`.

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture python_example -%}
# In this example, the lemma dictionary `lemmas_small.txt` has the form of
#
# ...
# pick	->	pick	picks	picking	picked
# peck	->	peck	pecking	pecked	pecks
# pickle	->	pickle	pickles	pickled	pickling
# pepper	->	pepper	peppers	peppered	peppering
# ...
#
# where each key is delimited by `->` and values are delimited by `\t`

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

lemmatizer = Lemmatizer() \
    .setInputCols(["token"]) \
    .setOutputCol("lemma") \
    .setDictionary("src/test/resources/lemma-corpus-small/lemmas_small.txt", "->", "\t")

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      sentenceDetector,
      tokenizer,
      lemmatizer
    ])

data = spark.createDataFrame([["Peter Pipers employees are picking pecks of pickled peppers."]]) \
    .toDF("text")

result = pipeline.fit(data).transform(data)
result.selectExpr("lemma.result").show(truncate=False)
+------------------------------------------------------------------+
|result                                                            |
+------------------------------------------------------------------+
|[Peter, Pipers, employees, are, pick, peck, of, pickle, pepper, .]|
+------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
// In this example, the lemma dictionary `lemmas_small.txt` has the form of
//
// ...
// pick	->	pick	picks	picking	picked
// peck	->	peck	pecking	pecked	pecks
// pickle	->	pickle	pickles	pickled	pickling
// pepper	->	pepper	peppers	peppered	peppering
// ...
//
// where each key is delimited by `->` and values are delimited by `\t`

import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Lemmatizer
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
  .setInputCols(Array("document"))
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols(Array("sentence"))
  .setOutputCol("token")

val lemmatizer = new Lemmatizer()
  .setInputCols(Array("token"))
  .setOutputCol("lemma")
  .setDictionary("src/test/resources/lemma-corpus-small/lemmas_small.txt", "->", "\t")

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    sentenceDetector,
    tokenizer,
    lemmatizer
  ))

val data = Seq("Peter Pipers employees are picking pecks of pickled peppers.")
  .toDF("text")

val result = pipeline.fit(data).transform(data)
result.selectExpr("lemma.result").show(false)
+------------------------------------------------------------------+
|result                                                            |
+------------------------------------------------------------------+
|[Peter, Pipers, employees, are, pick, peck, of, pickle, pepper, .]|
+------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[Lemmatizer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/Lemmatizer)
{%- endcapture -%}

{%- capture python_api_link -%}
[Lemmatizer](/api/python/reference/autosummary/python/sparknlp/annotator/lemmatizer/index.html#sparknlp.annotator.lemmatizer.Lemmatizer)
{%- endcapture -%}

{%- capture source_link -%}
[Lemmatizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Lemmatizer.scala)
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
