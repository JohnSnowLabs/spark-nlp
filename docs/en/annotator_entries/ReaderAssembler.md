{%- capture title -%}
ReaderAssembler
{%- endcapture -%}

{%- capture description -%}
The ReaderAssembler annotator provides a unified interface to combine multiple Spark NLP readers such as `Reader2Doc`, `Reader2Table`, and `Reader2Image` into a single, configurable component. It automatically selects and orchestrates the appropriate reader for each input based on file type, content type, and configured priorities, allowing you to process heterogeneous content (documents, tables, images) seamlessly in one pipeline.

Supported Input Types:
- Text: `txt`, `html`, `htm`, `md`, `xml`, `csv`  
- Documents: `pdf`, `doc`, `docx`, `xls`, `xlsx`, `ppt`, `pptx`  
- Email: `eml`, `msg`  
- Images: `png`, `jpg`, `jpeg`, `bmp`, `gif`  

Parameters:
- `contentPath`: Path to the content source (file or directory).  
- `inputCol`: Input column name for in-memory string content (optional).  
- `outputCol`: Base output column name; appends `text`, `table`, `image` for respective reader outputs (default: `document`).  
- `contentType`: MIME type of the content (e.g., `text/html`, `application/pdf`) (optional).  
- `explodeDocs`: Whether to split multi-document files into separate rows (default: `false`).  
- `flattenOutput`: Whether to return plain content with minimal metadata (default: `false`).  
- `inferTableStructure`: Whether to automatically detect table structure from tabular content (default: `true`).  
- `excludeNonText`: Whether to ignore non-text rows, such as tables, in Reader2Doc (default: `false`).  
- `userMessage`: Custom message describing the content for prompt-based models (op—ional).  
- `promptTemplate`: Template format for prompt generation (optional).  
- `customPromptTemplate`: Custom prompt template if `promptTemplate` is set to `"custom"` (optional).  

For an extended example see the
[example notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/examples/python/data-preprocessing/SparkNLP_ReaderAssembler_Demo.ipynb).

{%- endcapture -%}

{%- capture input_anno -%}
NONE
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT, TABLE, IMAGE
{%- endcapture -%}

{%- capture python_example -%}
from johnsnowlabs.reader import ReaderAssembler
from pyspark.ml import Pipeline

reader_assembler = ReaderAssembler() \
    .setContentType("text/html") \
    .setContentPath("/table-image.html") \
    .setOutputCol("document")

pipeline = Pipeline(stages=[reader_assembler])

pipeline_model = pipeline.fit(empty_data_set)
result_df = pipeline_model.transform(empty_data_set)

result_df.show()
+--------+--------------------+--------------------+--------------------+---------+
|fileName|       document_text|      document_table|      document_image|exception|
+--------+--------------------+--------------------+--------------------+---------+
|    null|[{document, 0, 26...|[{document, 0, 50...|[{image, , 5, 5, ...|     null|
+--------+--------------------+--------------------+--------------------+---------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.reader.ReaderAssembler
import org.apache.spark.ml.Pipeline

val readerAssembler = new ReaderAssembler()
  .setContentType("text/html")
  .setContentPath(s"$htmlFilesDirectory/table-image.html")
  .setOutputCol("document")

val pipeline = new Pipeline().setStages(Array(readerAssembler))

val pipelineModel = pipeline.fit(emptyDataSet)
val resultDf = pipelineModel.transform(emptyDataSet)

resultDf.show()
+--------+--------------------+--------------------+--------------------+---------+
|fileName|       document_text|      document_table|      document_image|exception|
+--------+--------------------+--------------------+--------------------+---------+
|    null|[{document, 0, 26...|[{document, 0, 50...|[{image, , 5, 5, ...|     null|
+--------+--------------------+--------------------+--------------------+---------+
{%- endcapture -%}

{%- capture api_link -%} 
[ReaderAssembler](/api/com/johnsnowlabs/reader/ReaderAssembler.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[ReaderAssembler](/api/python/reference/autosummary/sparknlp/reader/reader_assembler/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[ReaderAssembler](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/reader/ReaderAssembler.scala)
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