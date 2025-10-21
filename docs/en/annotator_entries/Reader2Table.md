{%- capture title -%}
Reader2Table
{%- endcapture -%}

{%- capture description -%}
The Reader2Table annotator enables seamless extraction of tabular content from documents within existing Spark NLP workflows. It allows you to efficiently parse tables from a wide variety of file types, including plain text, HTML, Word, Excel, PowerPoint, and CSV files, and return them as structured Spark DataFrames with metadata, ready for downstream processing or analysis.

Supported File Formats:
- Text files: `.txt`  
- HTML: `.html`, `.htm`  
- Word documents: `.doc`, `.docx`  
- Excel spreadsheets: `.xls`, `.xlsx`  
- PowerPoint presentations: `.ppt`, `.pptx`  
- CSV files: `.csv`  

Parameters:
- `contentPath`: Path to the input documents or table-containing files (required).  
- `inputCol`: Input column name (optional).  
- `outputCol`: Output column name for structured table data (default: `document`).  
- `contentType`: MIME type of the documents (e.g., `"text/html"`, `"application/vnd.ms-excel"`) (optional).  
- `explodeDocs`: Whether to split multi-document files into separate rows (default: `true`).  
- `storeContent`: Whether to include the raw file content alongside structured output (default: `false`).  
- `flattenOutput`: Whether to return plain table data with minimal metadata (default: `false`).  
- `inferTableStructure`: Whether to detect and preserve table structure from the input (default: `true`).  
- `outputFormat`: Format of the extracted table output (`json-table` or `html-table`, default: `json-table`).  
- `ignoreExceptions`: Whether to ignore exceptions during processing (default: `true`).  

For an extended example see the
[example notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/examples/python/data-preprocessing/SparkNLP_Reader2Table_Demo.ipynb).

{%- endcapture -%}

{%- capture input_anno -%}
NONE
{%- endcapture -%}

{%- capture output_anno -%}
TABLE
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.reader.reader2table import Reader2Table
from pyspark.ml import Pipeline

reader2Table = Reader2Table() \
    .setContentType("application/csv") \
    .setContentPath(f"./pdfDirectory")

pipeline = Pipeline(stages=[reader2Table])

pipelineModel = pipeline.fit(emptyDataSet)
resultDf = pipelineModel.transform(emptyDataSet)

resultDf.show()
+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|fileName        |document                                                                                                                                                                                    |
+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|stanley-cups.csv|[{document, 0, 137, {"caption":"","header":[],"rows":[["Team","Location","Stanley Cups"],["Blues","STL","1"],["Flyers","PHI","2"],["Maple Leafs","TOR","13"]]}, {elementType -> Table}, []}]|
+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.reader.Reader2Table
import org.apache.spark.ml.Pipeline

val reader2Table = new Reader2Table()
  .setContentType("application/csv")
  .setContentPath(s"$pdfDirectory/")

val pipeline = new Pipeline().setStages(Array(reader2Table))

val pipelineModel = pipeline.fit(emptyDataSet)
val resultDf = pipelineModel.transform(emptyDataSet)

resultDf.show()
+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|fileName        |document                                                                                                                                                                                    |
+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|stanley-cups.csv|[{document, 0, 137, {"caption":"","header":[],"rows":[["Team","Location","Stanley Cups"],["Blues","STL","1"],["Flyers","PHI","2"],["Maple Leafs","TOR","13"]]}, {elementType -> Table}, []}]|
+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture api_link -%} 
[Reader2Table](/api/com/johnsnowlabs/reader/Reader2Table.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[Reader2Table](/api/python/reference/autosummary/sparknlp/reader/Reader2Table/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[Reader2Table](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/reader/Reader2Table.scala)
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