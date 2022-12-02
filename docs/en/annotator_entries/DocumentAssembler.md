{%- capture title -%}
DocumentAssembler
{%- endcapture -%}

{%- capture description -%}
Prepares data into a format that is processable by Spark NLP. This is the entry point for every Spark NLP pipeline.
The `DocumentAssembler` can read either a `String` column or an `Array[String]`. Additionally, setCleanupMode
can be used to pre-process the text (Default: `disabled`). For possible options please refer the parameters section.

For more extended examples on document pre-processing see the
[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
NONE
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

data = spark.createDataFrame([["Spark NLP is an open-source text processing library."]]).toDF("text")
documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

result = documentAssembler.transform(data)

result.select("document").show(truncate=False)
+----------------------------------------------------------------------------------------------+
|document                                                                                      |
+----------------------------------------------------------------------------------------------+
|[[document, 0, 51, Spark NLP is an open-source text processing library., [sentence -> 0], []]]|
+----------------------------------------------------------------------------------------------+

result.select("document").printSchema
root
 |-- document: array (nullable = True)
 |    |-- element: struct (containsNull = True)
 |    |    |-- annotatorType: string (nullable = True)
 |    |    |-- begin: integer (nullable = False)
 |    |    |-- end: integer (nullable = False)
 |    |    |-- result: string (nullable = True)
 |    |    |-- metadata: map (nullable = True)
 |    |    |    |-- key: string
 |    |    |    |-- value: string (valueContainsNull = True)
 |    |    |-- embeddings: array (nullable = True)
 |    |    |    |-- element: float (containsNull = False)

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler

val data = Seq("Spark NLP is an open-source text processing library.").toDF("text")
val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")

val result = documentAssembler.transform(data)

result.select("document").show(false)
+----------------------------------------------------------------------------------------------+
|document                                                                                      |
+----------------------------------------------------------------------------------------------+
|[[document, 0, 51, Spark NLP is an open-source text processing library., [sentence -> 0], []]]|
+----------------------------------------------------------------------------------------------+

result.select("document").printSchema
root
 |-- document: array (nullable = true)
 |    |-- element: struct (containsNull = true)
 |    |    |-- annotatorType: string (nullable = true)
 |    |    |-- begin: integer (nullable = false)
 |    |    |-- end: integer (nullable = false)
 |    |    |-- result: string (nullable = true)
 |    |    |-- metadata: map (nullable = true)
 |    |    |    |-- key: string
 |    |    |    |-- value: string (valueContainsNull = true)
 |    |    |-- embeddings: array (nullable = true)
 |    |    |    |-- element: float (containsNull = false)

{%- endcapture -%}

{%- capture api_link -%}
[DocumentAssembler](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/DocumentAssembler)
{%- endcapture -%}

{%- capture python_api_link -%}
[DocumentAssembler](/api/python/reference/autosummary/python/sparknlp/base/document_assembler/index.html#sparknlp.base.document_assembler.DocumentAssembler)
{%- endcapture -%}

{%- capture source_link -%}
[DocumentAssembler](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/DocumentAssembler.scala)
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