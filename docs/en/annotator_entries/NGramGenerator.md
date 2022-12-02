{%- capture title -%}
NGramGenerator
{%- endcapture -%}

{%- capture description -%}
A feature transformer that converts the input array of strings (annotatorType TOKEN) into an
array of n-grams (annotatorType CHUNK).
Null values in the input array are ignored.
It returns an array of n-grams where each n-gram is represented by a space-separated string of
words.

When the input is empty, an empty array is returned.
When the input array length is less than n (number of elements per n-gram), no n-grams are
returned.

For more extended examples see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/chunking/NgramGenerator.ipynb)
and the [NGramGeneratorTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/NGramGeneratorTestSpec.scala).
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

nGrams = NGramGenerator() \
    .setInputCols(["token"]) \
    .setOutputCol("ngrams") \
    .setN(2)

pipeline = Pipeline().setStages([
      documentAssembler,
      sentence,
      tokenizer,
      nGrams
    ])

data = spark.createDataFrame([["This is my sentence."]]).toDF("text")
results = pipeline.fit(data).transform(data)

results.selectExpr("explode(ngrams) as result").show(truncate=False)
+------------------------------------------------------------+
|result                                                      |
+------------------------------------------------------------+
|[chunk, 0, 6, This is, [sentence -> 0, chunk -> 0], []]     |
|[chunk, 5, 9, is my, [sentence -> 0, chunk -> 1], []]       |
|[chunk, 8, 18, my sentence, [sentence -> 0, chunk -> 2], []]|
|[chunk, 11, 19, sentence ., [sentence -> 0, chunk -> 3], []]|
+------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.NGramGenerator
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols(Array("sentence"))
  .setOutputCol("token")

val nGrams = new NGramGenerator()
  .setInputCols("token")
  .setOutputCol("ngrams")
  .setN(2)

val pipeline = new Pipeline().setStages(Array(
    documentAssembler,
    sentence,
    tokenizer,
    nGrams
  ))

val data = Seq("This is my sentence.").toDF("text")
val results = pipeline.fit(data).transform(data)

results.selectExpr("explode(ngrams) as result").show(false)
+------------------------------------------------------------+
|result                                                      |
+------------------------------------------------------------+
|[chunk, 0, 6, This is, [sentence -> 0, chunk -> 0], []]     |
|[chunk, 5, 9, is my, [sentence -> 0, chunk -> 1], []]       |
|[chunk, 8, 18, my sentence, [sentence -> 0, chunk -> 2], []]|
|[chunk, 11, 19, sentence ., [sentence -> 0, chunk -> 3], []]|
+------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[NGramGenerator](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/NGramGenerator)
{%- endcapture -%}

{%- capture python_api_link -%}
[NGramGenerator](/api/python/reference/autosummary/python/sparknlp/annotator/n_gram_generator/index.html#sparknlp.annotator.n_gram_generator.NGramGenerator)
{%- endcapture -%}

{%- capture source_link -%}
[NGramGenerator](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/NGramGenerator.scala)
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