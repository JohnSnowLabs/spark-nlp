{%- capture title -%}
Stemmer
{%- endcapture -%}

{%- capture description -%}
Returns hard-stems out of words with the objective of retrieving the meaningful part of the word.
For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture python_example -%}
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

stemmer = Stemmer() \
    .setInputCols(["token"]) \
    .setOutputCol("stem")

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    stemmer
])

data = spark.createDataFrame([["Peter Pipers employees are picking pecks of pickled peppers."]]) \
    .toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("stem.result").show(truncate = False)
+-------------------------------------------------------------+
|result                                                       |
+-------------------------------------------------------------+
|[peter, piper, employe, ar, pick, peck, of, pickl, pepper, .]|
+-------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{Stemmer, Tokenizer}
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val stemmer = new Stemmer()
  .setInputCols("token")
  .setOutputCol("stem")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  stemmer
))

val data = Seq("Peter Pipers employees are picking pecks of pickled peppers.")
  .toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("stem.result").show(truncate = false)
+-------------------------------------------------------------+
|result                                                       |
+-------------------------------------------------------------+
|[peter, piper, employe, ar, pick, peck, of, pickl, pepper, .]|
+-------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[Stemmer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/Stemmer)
{%- endcapture -%}

{%- capture python_api_link -%}
[Stemmer](/api/python/reference/autosummary/python/sparknlp/annotator/stemmer/index.html#sparknlp.annotator.stemmer.Stemmer)
{%- endcapture -%}

{%- capture source_link -%}
[Stemmer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Stemmer.scala)
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