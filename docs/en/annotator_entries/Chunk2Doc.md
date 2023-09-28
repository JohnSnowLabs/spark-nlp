{%- capture title -%}
Chunk2Doc
{%- endcapture -%}

{%- capture description -%}
Converts a `CHUNK` type column back into `DOCUMENT`. Useful when trying to re-tokenize or do further analysis on a
`CHUNK` result.
{%- endcapture -%}

{%- capture input_anno -%}
CHUNK
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from sparknlp.pretrained import PretrainedPipeline
# Location entities are extracted and converted back into `DOCUMENT` type for further processing

data = spark.createDataFrame([[1, "New York and New Jersey aren't that far apart actually."]]).toDF("id", "text")

# Extracts Named Entities amongst other things
pipeline = PretrainedPipeline("explain_document_dl")

chunkToDoc = Chunk2Doc().setInputCols("entities").setOutputCol("chunkConverted")
explainResult = pipeline.transform(data)

result = chunkToDoc.transform(explainResult)
result.selectExpr("explode(chunkConverted)").show(truncate=False)
+------------------------------------------------------------------------------+
|col                                                                           |
+------------------------------------------------------------------------------+
|[document, 0, 7, New York, [entity -> LOC, sentence -> 0, chunk -> 0], []]    |
|[document, 13, 22, New Jersey, [entity -> LOC, sentence -> 0, chunk -> 1], []]|
+------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
// Location entities are extracted and converted back into `DOCUMENT` type for further processing
import spark.implicits._
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.Chunk2Doc

val data = Seq((1, "New York and New Jersey aren't that far apart actually.")).toDF("id", "text")

// Extracts Named Entities amongst other things
val pipeline = PretrainedPipeline("explain_document_dl")

val chunkToDoc = new Chunk2Doc().setInputCols("entities").setOutputCol("chunkConverted")
val explainResult = pipeline.transform(data)

val result = chunkToDoc.transform(explainResult)
result.selectExpr("explode(chunkConverted)").show(false)
+------------------------------------------------------------------------------+
|col                                                                           |
+------------------------------------------------------------------------------+
|[document, 0, 7, New York, [entity -> LOC, sentence -> 0, chunk -> 0], []]    |
|[document, 13, 22, New Jersey, [entity -> LOC, sentence -> 0, chunk -> 1], []]|
+------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[Chunk2Doc](/api/com/johnsnowlabs/nlp/annotators/Chunk2Doc)
{%- endcapture -%}

{%- capture python_api_link -%}
[Chunk2Doc](/api/python/reference/autosummary/sparknlp/annotator/chunk2_doc/index.html#sparknlp.base.chunk2_doc.Chunk2Doc)
{%- endcapture -%}

{%- capture source_link -%}
[Chunk2Doc](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/Chunk2Doc.scala)
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