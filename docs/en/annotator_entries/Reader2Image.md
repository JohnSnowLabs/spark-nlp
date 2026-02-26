{%- capture title -%}
Reader2Image
{%- endcapture -%}

{%- capture description -%}
The Reader2Image annotator enables seamless integration of image reading capabilities into existing Spark NLP workflows. It allows you to efficiently extract and structure image content from both individual image files and documents with embedded images.

With this, you can read image files or extract images from documents such as PDFs, Word, Excel, PowerPoint, HTML, Markdown, and email files. All extracted images are returned as structured Spark DataFrames with associated metadata, ready for downstream processing in Spark NLP pipelines.

Supported File Formats:
- Image files: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`  
- Documents with embedded images: `.pdf`, `.doc`, `.docx`, `.ppt`, `.pptx`, `.xls`, `.xlsx`, `.eml`, `.msg`, `.html`, `.htm`, `.md`

Parameters:
- `contentPath`: Path to the documents or image files to read.  
- `inputCol`: Input column containing the documents.  
- `outputCol`: Output column for structured image data (default: `"image"`).  
- `explodeDocs`: Whether to split multi-document files into separate rows (default: `true`).  
- `explodeImages`: Whether to output one image per row (default: `true`). Set to `false` to combine multiple images per file into a single record.  
- `readAsImage`: Whether to read PDF pages as images (default: `true`).  
- `storeContent`: Whether to store the raw file content alongside structured output (default: `false`).  
- `flattenOutput`: Whether to return plain image data with minimal metadata (default: `false`).  
- `outputAsDocument`: Whether to output all extracted images as a single combined document (default: `false`).  
- `excludeNonImage`: Whether to skip non-image content found in mixed-format files (default: `false`).  
- `contentType`: MIME type of the documents (e.g., "text/html", "application/pdf").  
- `userMessage`: Custom message describing the image for prompt-based models (default: `"Describe this image"`).  
- `promptTemplate`: Format of the output prompt for image models (default: `"qwen2vl-chat"`).  
- `customPromptTemplate`: Custom prompt template for image models when `promptTemplate` is `"custom"`.

For an extended example see the
[example notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/examples/python/data-preprocessing/SparkNLP_Reader2Image_Demo.ipynb).

{%- endcapture -%}

{%- capture input_anno -%}
NONE
{%- endcapture -%}

{%- capture output_anno -%}
IMAGE
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.reader.reader2image import Reader2Image
from pyspark.ml import Pipeline

reader2Image = Reader2Image() \
    .setContentType("text/html") \
    .setContentPath("./example-images.html") \
    .setOutputCol("image")

pipeline = Pipeline(stages=[reader2Image])

pipelineModel = pipeline.fit(emptyDataSet)
resultDf = pipelineModel.transform(emptyDataSet)

resultDf.show()
+-------------------+--------------------+
|           fileName|               image|
+-------------------+--------------------+
|example-images.html|[{image, example-...|
|example-images.html|[{image, example-...|
+-------------------+--------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.reader.Reader2Image
import com.johnsnowlabs.nlp.base.DocumentAssembler
import org.apache.spark.ml.Pipeline

val reader2Image = new Reader2Image()
  .setContentType("text/html")
  .setContentPath("./example-images.html")
  .setOutputCol("image")

val pipeline = new Pipeline().setStages(Array(reader2Image))

val pipelineModel = pipeline.fit(emptyDataSet)
val resultDf = pipelineModel.transform(emptyDataSet)

resultDf.show()
+-------------------+--------------------+
|           fileName|               image|
+-------------------+--------------------+
|example-images.html|[{image, example-...|
|example-images.html|[{image, example-...|
+-------------------+--------------------+
{%- endcapture -%}

{%- capture api_link -%} 
[Reader2Image](/api/com/johnsnowlabs/reader/Reader2Image.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[Reader2Image](/api/python/reference/autosummary/sparknlp/reader/Reader2Image/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[Reader2Image](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/reader/Reader2Image.scala)
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