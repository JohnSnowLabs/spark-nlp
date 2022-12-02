{%- capture title -%}
Lemmatizer
{%- endcapture -%}

{%- capture model_description -%}
Instantiated Model of the Lemmatizer. For usage and examples, please see the documentation of that class.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Lemmatization).
{%- endcapture -%}

{%- capture model_input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
# The lemmatizer from the example of the [[Lemmatizer]] can be replaced with:
lemmatizer = LemmatizerModel.pretrained() \
    .setInputCols(["token"]) \
    .setOutputCol("lemma")

{%- endcapture -%}

{%- capture model_scala_example -%}
// The lemmatizer from the example of the [[Lemmatizer]] can be replaced with:
val lemmatizer = LemmatizerModel.pretrained()
  .setInputCols(Array("token"))
  .setOutputCol("lemma")

{%- endcapture -%}

{%- capture model_api_link -%}
[LemmatizerModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/LemmatizerModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[LemmatizerModel](/api/python/reference/autosummary/python/sparknlp/annotator/lemmatizer/index.html#sparknlp.annotator.lemmatizer.LemmatizerModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[LemmatizerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/LemmatizerModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Class to find lemmas out of words with the objective of returning a base dictionary word.
Retrieves the significant part of a word. A dictionary of predefined lemmas must be provided with `setDictionary`.
The dictionary can be set as a delimited text file.
Pretrained models can be loaded with `LemmatizerModel.pretrained`.

For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Lemmatization).
For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb).
{%- endcapture -%}

{%- capture approach_input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture approach_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture approach_python_example -%}
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

{%- capture approach_scala_example -%}
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

{%- capture approach_api_link -%}
[Lemmatizer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/Lemmatizer)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[Lemmatizer](/api/python/reference/autosummary/python/sparknlp/annotator/lemmatizer/index.html#sparknlp.annotator.lemmatizer.Lemmatizer)
{%- endcapture -%}

{%- capture approach_source_link -%}
[Lemmatizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Lemmatizer.scala)
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
