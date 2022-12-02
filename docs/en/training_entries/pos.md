{%- capture title -%}
POS Dataset
{%- endcapture -%}

{%- capture description -%}
In order to train a Part of Speech Tagger annotator, we need to get corpus data as a Spark dataframe. There is a component that does this for us: it reads a plain text file and transforms it to a Spark dataset.
{%- endcapture -%}

{%- capture file_format -%}
A|DT few|JJ months|NNS ago|RB you|PRP received|VBD a|DT letter|NN
{%- endcapture -%}

{%- capture constructor -%}
None
{%- endcapture -%}

{%- capture read_dataset_params -%}
- **spark**: Initiated Spark Session with Spark NLP
- **path**: Path to the resource
- **delimiter**: Delimiter of word and POS, by default "\|"
- **outputPosCol**: Name of the output POS column, by default “tags”
- **outputDocumentCol**: Name of the output document column, by default “document”
- **outputTextCol**: Name of the output text column, by default “text”
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.training import POS
pos = POS()
path = "src/test/resources/anc-pos-corpus-small/test-training.txt"
posDf = pos.readDataset(spark, path, "|", "tags")
posDf.selectExpr("explode(tags) as tags").show(truncate=False)
+---------------------------------------------+
|tags                                         |
+---------------------------------------------+
|[pos, 0, 5, NNP, [word -> Pierre], []]       |
|[pos, 7, 12, NNP, [word -> Vinken], []]      |
|[pos, 14, 14, ,, [word -> ,], []]            |
|[pos, 16, 17, CD, [word -> 61], []]          |
|[pos, 19, 23, NNS, [word -> years], []]      |
|[pos, 25, 27, JJ, [word -> old], []]         |
|[pos, 29, 29, ,, [word -> ,], []]            |
|[pos, 31, 34, MD, [word -> will], []]        |
|[pos, 36, 39, VB, [word -> join], []]        |
|[pos, 41, 43, DT, [word -> the], []]         |
|[pos, 45, 49, NN, [word -> board], []]       |
|[pos, 51, 52, IN, [word -> as], []]          |
|[pos, 47, 47, DT, [word -> a], []]           |
|[pos, 56, 67, JJ, [word -> nonexecutive], []]|
|[pos, 69, 76, NN, [word -> director], []]    |
|[pos, 78, 81, NNP, [word -> Nov.], []]       |
|[pos, 83, 84, CD, [word -> 29], []]          |
|[pos, 81, 81, ., [word -> .], []]            |
+---------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.training.POS

val pos = POS()
val path = "src/test/resources/anc-pos-corpus-small/test-training.txt"
val posDf = pos.readDataset(spark, path, "|", "tags")

posDf.selectExpr("explode(tags) as tags").show(false)
+---------------------------------------------+
|tags                                         |
+---------------------------------------------+
|[pos, 0, 5, NNP, [word -> Pierre], []]       |
|[pos, 7, 12, NNP, [word -> Vinken], []]      |
|[pos, 14, 14, ,, [word -> ,], []]            |
|[pos, 16, 17, CD, [word -> 61], []]          |
|[pos, 19, 23, NNS, [word -> years], []]      |
|[pos, 25, 27, JJ, [word -> old], []]         |
|[pos, 29, 29, ,, [word -> ,], []]            |
|[pos, 31, 34, MD, [word -> will], []]        |
|[pos, 36, 39, VB, [word -> join], []]        |
|[pos, 41, 43, DT, [word -> the], []]         |
|[pos, 45, 49, NN, [word -> board], []]       |
|[pos, 51, 52, IN, [word -> as], []]          |
|[pos, 47, 47, DT, [word -> a], []]           |
|[pos, 56, 67, JJ, [word -> nonexecutive], []]|
|[pos, 69, 76, NN, [word -> director], []]    |
|[pos, 78, 81, NNP, [word -> Nov.], []]       |
|[pos, 83, 84, CD, [word -> 29], []]          |
|[pos, 81, 81, ., [word -> .], []]            |
+---------------------------------------------+
{%- endcapture -%}

{%- capture api_link -%}
[POS](/api/com/johnsnowlabs/nlp/training/POS.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[POS](/api/python/reference/autosummary/python/sparknlp/training/pos/index.html#sparknlp.training.pos.POS)
{%- endcapture -%}

{%- capture source_link -%}
[POS.scala](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/training/POS.scala)
{%- endcapture -%}

{% include templates/training_dataset_entry.md
title=title
description=description
file_format=file_format
constructor=constructor
read_dataset_params=read_dataset_params
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
source_link=source_link
%}