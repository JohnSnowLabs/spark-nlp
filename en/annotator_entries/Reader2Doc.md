{%- capture title -%}
Reader2Doc
{%- endcapture -%}

{%- capture description -%}
The Reader2Doc annotator enables seamless integration of document reading capabilities into existing Spark NLP workflows. It allows you to efficiently extract and structure content from a wide range of document types, making it easier to reuse and extend your pipelines without additional preprocessing steps.

Supported File Formats:
- Text: `.txt`  
- HTML: `.html`, `.htm`  
- Microsoft Word: `.doc`, `.docx`  
- Microsoft Excel: `.xls`, `.xlsx`  
- Microsoft PowerPoint: `.ppt`, `.pptx`  
- Email files: `.eml`, `.msg`  
- PDF documents: `.pdf` 

Parameters:
- `explodeDocs` : Whether to output one document per row (default: `true`). Set to `false` to combine all content into a single row per input file.  
- `flattenOutput` : Whether to output plain text with minimal metadata (default: `false`).  
- `outputAsDocument` : Whether to output data as one single document instead of multiple records (default: `false`).  
- `excludeNonText` : Whether to exclude non-textual data such as tables and images (default: `false`).  

For an extended example see the
[example notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/examples/python/data-preprocessing/SparkNLP_Reader2Doc_Demo.ipynb).

{%- endcapture -%}

{%- capture input_anno -%}
NONE
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
from johnsnowlabs.reader import Reader2Doc
from johnsnowlabs.nlp.base import DocumentAssembler
from pyspark.ml import Pipeline

reader2doc = Reader2Doc() \
    .setContentType("application/pdf") \
    .setContentPath(f"{pdf_directory}/")

pipeline = Pipeline(stages=[reader2doc])

pipeline_model = pipeline.fit(empty_data_set)
result_df = pipeline_model.transform(empty_data_set)

result_df.show()
+------------------------------------------------------------------------------------------------------------------------------------+
|document                                                                                                                            |
+------------------------------------------------------------------------------------------------------------------------------------+
|[{document, 0, 14, This is a Title, {pageNumber -> 1, elementType -> Title, fileName -> pdf-title.pdf}, []}]                        |
|[{document, 15, 38, This is a narrative text, {pageNumber -> 1, elementType -> NarrativeText, fileName -> pdf-title.pdf}, []}]      |
|[{document, 39, 68, This is another narrative text, {pageNumber -> 1, elementType -> NarrativeText, fileName -> pdf-title.pdf}, []}]|
+------------------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.reader.Reader2Doc
import com. johnsnowlabs.nlp.base.DocumentAssembler
import org.apache.spark.ml.Pipeline

val reader2Doc = new Reader2Doc()
  .setContentType("application/pdf")
  .setContentPath(s"$pdfDirectory/")

val pipeline = new Pipeline()
  .setStages(Array(reader2Doc))

val pipelineModel = pipeline.fit(emptyDataSet)
val resultDf = pipelineModel.transform(emptyDataSet)

resultDf.show()
+------------------------------------------------------------------------------------------------------------------------------------+
|document                                                                                                                            |
+------------------------------------------------------------------------------------------------------------------------------------+
|[{document, 0, 14, This is a Title, {pageNumber -> 1, elementType -> Title, fileName -> pdf-title.pdf}, []}]                        |
|[{document, 15, 38, This is a narrative text, {pageNumber -> 1, elementType -> NarrativeText, fileName -> pdf-title.pdf}, []}]      |
|[{document, 39, 68, This is another narrative text, {pageNumber -> 1, elementType -> NarrativeText, fileName -> pdf-title.pdf}, []}]|
+------------------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture api_link -%} 
[Reader2Doc](/api/com/johnsnowlabs/reader/Reader2Doc.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[Reader2Doc](/api/python/reference/autosummary/sparknlp/reader/reader2doc/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[Reader2Doc](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/reader/Reader2Doc.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}