{%- capture title -%}
NerConverter
{%- endcapture -%}

{%- capture description -%}
Converts a IOB or IOB2 representation of NER to a user-friendly one,
by associating the tokens of recognized entities and their label. Results in `CHUNK` Annotation type.

NER chunks can then be filtered by setting a whitelist with `setWhiteList`.
Chunks with no associated entity (tagged "O") are filtered.

See also [Inside–outside–beginning (tagging)](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) for more information.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN, NAMED_ENTITY
{%- endcapture -%}

{%- capture output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
# This is a continuation of the example of the NerDLModel. See that class
# on how to extract the entities.
# The output of the NerDLModel follows the Annotator schema and can be converted like so:
#
# result.selectExpr("explode(ner)").show(truncate=False)
# +----------------------------------------------------+
# |col                                                 |
# +----------------------------------------------------+
# |[named_entity, 0, 2, B-ORG, [word -> U.N], []]      |
# |[named_entity, 3, 3, O, [word -> .], []]            |
# |[named_entity, 5, 12, O, [word -> official], []]    |
# |[named_entity, 14, 18, B-PER, [word -> Ekeus], []]  |
# |[named_entity, 20, 24, O, [word -> heads], []]      |
# |[named_entity, 26, 28, O, [word -> for], []]        |
# |[named_entity, 30, 36, B-LOC, [word -> Baghdad], []]|
# |[named_entity, 37, 37, O, [word -> .], []]          |
# +----------------------------------------------------+
#
# After the converter is used:
converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("entities")

converter.transform(result).selectExpr("explode(entities)").show(truncate=False)
+------------------------------------------------------------------------+
|col                                                                     |
+------------------------------------------------------------------------+
|[chunk, 0, 2, U.N, [entity -> ORG, sentence -> 0, chunk -> 0], []]      |
|[chunk, 14, 18, Ekeus, [entity -> PER, sentence -> 0, chunk -> 1], []]  |
|[chunk, 30, 36, Baghdad, [entity -> LOC, sentence -> 0, chunk -> 2], []]|
+------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
// This is a continuation of the example of the [[com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel NerDLModel]]. See that class
// on how to extract the entities.
// The output of the NerDLModel follows the Annotator schema and can be converted like so:
//
// result.selectExpr("explode(ner)").show(false)
// +----------------------------------------------------+
// |col                                                 |
// +----------------------------------------------------+
// |[named_entity, 0, 2, B-ORG, [word -> U.N], []]      |
// |[named_entity, 3, 3, O, [word -> .], []]            |
// |[named_entity, 5, 12, O, [word -> official], []]    |
// |[named_entity, 14, 18, B-PER, [word -> Ekeus], []]  |
// |[named_entity, 20, 24, O, [word -> heads], []]      |
// |[named_entity, 26, 28, O, [word -> for], []]        |
// |[named_entity, 30, 36, B-LOC, [word -> Baghdad], []]|
// |[named_entity, 37, 37, O, [word -> .], []]          |
// +----------------------------------------------------+
//
// After the converter is used:
val converter = new NerConverter()
  .setInputCols("sentence", "token", "ner")
  .setOutputCol("entities")
  .setPreservePosition(false)

converter.transform(result).selectExpr("explode(entities)").show(false)
+------------------------------------------------------------------------+
|col                                                                     |
+------------------------------------------------------------------------+
|[chunk, 0, 2, U.N, [entity -> ORG, sentence -> 0, chunk -> 0], []]      |
|[chunk, 14, 18, Ekeus, [entity -> PER, sentence -> 0, chunk -> 1], []]  |
|[chunk, 30, 36, Baghdad, [entity -> LOC, sentence -> 0, chunk -> 2], []]|
+------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[NerConverter](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ner/NerConverter)
{%- endcapture -%}

{%- capture python_api_link -%}
[NerConverter](/api/python/reference/autosummary/python/sparknlp/annotator/ner/ner_converter/index.html#sparknlp.annotator.ner.ner_converter.NerConverter)
{%- endcapture -%}

{%- capture source_link -%}
[NerConverter](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/NerConverter.scala)
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