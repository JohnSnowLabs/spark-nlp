{%- capture title -%}
DocumentTitleSplitter
{%- endcapture -%}

{%- capture description -%}
Annotator that groups element-level documents into title-aware sections.

`DocumentTitleSplitter` is intended to work with element-level `DOCUMENT` annotations,
typically produced by `Reader2Doc().setOutputAsDocument(false).setExplodeDocs(false)`.
Whenever an input annotation has `metadata["elementType"] == "Title"`, it starts a new
semantic section and the title stays with the following content.

If overflow splitting is enabled, sections larger than `maxCharacters` are split into
smaller documents. Otherwise, each semantic section is emitted as a single chunk regardless
of length.

For example, given element-level input annotations:

```python
Title -> "Overview"
NarrativeText -> "Intro paragraph."
Title -> "Configuration"
NarrativeText -> "Settings paragraph."
```

The resulting document chunks are:

```python
["Overview Intro paragraph.", "Configuration Settings paragraph."]
```

Additionally, you can set

- the join string used between elements with `setJoinString`
- whether to split on page changes with `setSplitOnPageChange`
- whether to enable overflow splitting with `setEnableOverflowSplitting`
- the overflow chunk size with `setMaxCharacters`
- whether to explode the splits to individual rows with `setExplodeSplits`

For extended examples of usage, see the
[DocumentTitleSplitterTest](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/DocumentTitleSplitterTest.scala).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
from pyspark.ml import Pipeline

from sparknlp.annotator import DocumentTitleSplitter
from sparknlp.reader.reader2doc import Reader2Doc

empty_df = spark.createDataFrame([], "string").toDF("text")

reader2doc = Reader2Doc() \
    .setContentType("text/markdown") \
    .setContentPath("src/test/resources/reader/md/title-chunking.md") \
    .setOutputCol("document") \
    .setOutputAsDocument(False) \
    .setExplodeDocs(False)

titleSplitter = DocumentTitleSplitter() \
    .setInputCols(["document"]) \
    .setOutputCol("splits") \
    .setExplodeSplits(True)

pipeline = Pipeline().setStages([reader2doc, titleSplitter])
result = pipeline.fit(empty_df).transform(empty_df)

result.selectExpr(
      "splits.result as result",
      "splits.metadata.sectionTitle as sectionTitle") \
    .show(truncate = 80)
+--------------------------------------------------------------------------------+-------------+
|                                                                          result| sectionTitle|
+--------------------------------------------------------------------------------+-------------+
|[Overview Unstructured can parse Markdown into elements. This makes it easy t...|     Overview|
|[Configuration max_characters controls hard chunk size. new_after_n_chars con...|Configuration|
|[Example When the parser detects headings, they become Title elements. The ch...|      Example|
+--------------------------------------------------------------------------------+-------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.reader.Reader2Doc
import org.apache.spark.ml.Pipeline

val emptyDataSet = spark.createDataFrame(Seq.empty[String].map(Tuple1(_))).toDF("text")

val reader2Doc = new Reader2Doc()
  .setContentType("text/markdown")
  .setContentPath("src/test/resources/reader/md/title-chunking.md")
  .setOutputCol("document")
  .setOutputAsDocument(false)
  .setExplodeDocs(false)

val titleSplitter = new DocumentTitleSplitter()
  .setInputCols("document")
  .setOutputCol("splits")
  .setExplodeSplits(true)

val pipeline = new Pipeline().setStages(Array(reader2Doc, titleSplitter))
val result = pipeline.fit(emptyDataSet).transform(emptyDataSet)

result
  .selectExpr(
    "splits.result as result",
    "splits.metadata.sectionTitle as sectionTitle")
  .show(truncate = 80)
+--------------------------------------------------------------------------------+-------------+
|                                                                          result| sectionTitle|
+--------------------------------------------------------------------------------+-------------+
|[Overview Unstructured can parse Markdown into elements. This makes it easy t...|     Overview|
|[Configuration max_characters controls hard chunk size. new_after_n_chars con...|Configuration|
|[Example When the parser detects headings, they become Title elements. The ch...|      Example|
+--------------------------------------------------------------------------------+-------------+

{%- endcapture -%}

{%- capture api_link -%}
[DocumentTitleSplitter](/api/com/johnsnowlabs/nlp/annotators/DocumentTitleSplitter)
{%- endcapture -%}

{%- capture python_api_link -%}
[DocumentTitleSplitter](/api/python/reference/autosummary/sparknlp/annotator/document_title_splitter/index.html#sparknlp.annotator.document_title_splitter.DocumentTitleSplitter)
{%- endcapture -%}

{%- capture source_link -%}
[DocumentTitleSplitter](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/DocumentTitleSplitter.scala)
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
