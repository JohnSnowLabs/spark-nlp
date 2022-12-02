{%- capture title -%}
TokenAssembler
{%- endcapture -%}

{%- capture description -%}
This transformer reconstructs a `DOCUMENT` type annotation from tokens, usually after these have been normalized,
lemmatized, normalized, spell checked, etc, in order to use this document annotation in further annotators.
Requires `DOCUMENT` and `TOKEN` type annotations as input.

For more extended examples on document pre-processing see the
[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# First, the text is tokenized and cleaned
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentences")

tokenizer = Tokenizer() \
    .setInputCols(["sentences"]) \
    .setOutputCol("token")

normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized") \
    .setLowercase(False)

stopwordsCleaner = StopWordsCleaner() \
    .setInputCols(["normalized"]) \
    .setOutputCol("cleanTokens") \
    .setCaseSensitive(False)

# Then the TokenAssembler turns the cleaned tokens into a `DOCUMENT` type structure.
tokenAssembler = TokenAssembler() \
    .setInputCols(["sentences", "cleanTokens"]) \
    .setOutputCol("cleanText")

data = spark.createDataFrame([["Spark NLP is an open-source text processing library for advanced natural language processing."]]) \
    .toDF("text")

pipeline = Pipeline().setStages([
    documentAssembler,
    sentenceDetector,
    tokenizer,
    normalizer,
    stopwordsCleaner,
    tokenAssembler
]).fit(data)

result = pipeline.transform(data)
result.select("cleanText").show(truncate=False)
+---------------------------------------------------------------------------------------------------------------------------+
|cleanText                                                                                                                  |
+---------------------------------------------------------------------------------------------------------------------------+
|0, 80, Spark NLP opensource text processing library advanced natural language processing, [sentence -> 0], []|
+---------------------------------------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.annotator.{Normalizer, StopWordsCleaner}
import com.johnsnowlabs.nlp.TokenAssembler
import org.apache.spark.ml.Pipeline

// First, the text is tokenized and cleaned
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentences")

val tokenizer = new Tokenizer()
  .setInputCols("sentences")
  .setOutputCol("token")

val normalizer = new Normalizer()
  .setInputCols("token")
  .setOutputCol("normalized")
  .setLowercase(false)

val stopwordsCleaner = new StopWordsCleaner()
  .setInputCols("normalized")
  .setOutputCol("cleanTokens")
  .setCaseSensitive(false)

// Then the TokenAssembler turns the cleaned tokens into a `DOCUMENT` type structure.
val tokenAssembler = new TokenAssembler()
  .setInputCols("sentences", "cleanTokens")
  .setOutputCol("cleanText")

val data = Seq("Spark NLP is an open-source text processing library for advanced natural language processing.")
  .toDF("text")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  normalizer,
  stopwordsCleaner,
  tokenAssembler
)).fit(data)

val result = pipeline.transform(data)
result.select("cleanText").show(false)
+---------------------------------------------------------------------------------------------------------------------------+
|cleanText                                                                                                                  |
+---------------------------------------------------------------------------------------------------------------------------+
|[[document, 0, 80, Spark NLP opensource text processing library advanced natural language processing, [sentence -> 0], []]]|
+---------------------------------------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[TokenAssembler](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/TokenAssembler)
{%- endcapture -%}

{%- capture python_api_link -%}
[TokenAssembler](/api/python/reference/autosummary/python/sparknlp/base/token_assembler/index.html#sparknlp.base.token_assembler.TokenAssembler)
{%- endcapture -%}

{%- capture source_link -%}
[TokenAssembler](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/TokenAssembler.scala)
{%- endcapture -%}

{% include templates/anno_template.md
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