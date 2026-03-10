{%- capture title -%}
MultiColumnAssembler
{%- endcapture -%}

{%- capture description -%}
Merges multiple annotation columns into a single annotation column. This is useful when
multiple annotators produce separate annotation columns (e.g., `document_text`,
`document_table` from [ReaderAssembler](/docs/en/annotators#readerassembler)) and a downstream
annotator (e.g., [AutoGGUFVisionModel](/docs/en/annotators#autoggufvisionmodel)) expects a
single input column containing all annotations.

Annotations from all input columns are collected and concatenated into the output column.
The output annotator type defaults to `DOCUMENT` but can be configured via
`setOutputAsAnnotatorType`. Each annotation's metadata is preserved, and a `source_column`
key is added to track which input column the annotation originated from. All
annotations from the first input column appear first, then all from the second, and so on.

**Note:** All input columns must use the standard `Annotation` schema. Columns that use the
`AnnotationImage` schema (e.g., IMAGE-typed columns from `ReaderAssembler`) are **not
supported** and will cause a validation error.

For more extended examples see the
[Examples](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/annotation-merger/Merging_Annotation_Columns.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from pyspark.ml import Pipeline

documentAssembler1 = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document_text")

documentAssembler2 = DocumentAssembler() \
    .setInputCol("table") \
    .setOutputCol("document_table")

multiColumnAssembler = MultiColumnAssembler() \
    .setInputCols(["document_text", "document_table"]) \
    .setOutputCol("merged_document")

data = spark.createDataFrame(
    [("Hello world", "Name | Age\nJohn | 30")],
    ["text", "table"]
)

pipeline = Pipeline().setStages([
    documentAssembler1,
    documentAssembler2,
    multiColumnAssembler
]).fit(data)

result = pipeline.transform(data)
result.selectExpr("merged_document.result").show(truncate=False)
+--------------------------------+
|result                          |
+--------------------------------+
|[Hello world, Name | Age       |
|John | 30]                      |
+--------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.{MultiColumnAssembler, DocumentAssembler}
import org.apache.spark.ml.Pipeline

val documentAssembler1 = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document_text")

val documentAssembler2 = new DocumentAssembler()
  .setInputCol("table")
  .setOutputCol("document_table")

val multiColumnAssembler = new MultiColumnAssembler()
  .setInputCols("document_text", "document_table")
  .setOutputCol("merged_document")

val data = Seq(("Hello world", "Name | Age\nJohn | 30"))
  .toDF("text", "table")

val pipeline = new Pipeline()
  .setStages(Array(documentAssembler1, documentAssembler2, multiColumnAssembler))
  .fit(data)

val result = pipeline.transform(data)
result.selectExpr("merged_document.result").show(false)
+--------------------------------+
|result                          |
+--------------------------------+
|[Hello world, Name | Age       |
|John | 30]                      |
+--------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[MultiColumnAssembler](/api/com/johnsnowlabs/nlp/MultiColumnAssembler)
{%- endcapture -%}

{%- capture python_api_link -%}
[MultiColumnAssembler](/api/python/reference/autosummary/sparknlp/base/multi_column_assembler/index.html#sparknlp.base.multi_column_assembler.MultiColumnAssembler)
{%- endcapture -%}

{%- capture source_link -%}
[MultiColumnAssembler](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/MultiColumnAssembler.scala)
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

