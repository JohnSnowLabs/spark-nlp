{%- capture title -%}
NerConverterInternal
{%- endcapture -%}

{%- capture description -%}
Converts a IOB or IOB2 representation of NER to a user-friendly one,
by associating the tokens of recognized entities and their label.
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
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
# The output of a MedicalNerModel follows the Annotator schema and looks like this after the transformation.
result.selectExpr("explode(ner_result)").show(5, False)
+--------------------------------------------------------------------------+
|col                                                                       |
+--------------------------------------------------------------------------+
|{named_entity, 3, 3, O, {word -> A, confidence -> 0.994}, []}             |
|{named_entity, 5, 15, B-Age, {word -> 63-year-old, confidence -> 1.0}, []}|
|{named_entity, 17, 19, B-Gender, {word -> man, confidence -> 0.9858}, []} |
|{named_entity, 21, 28, O, {word -> presents, confidence -> 0.9952}, []}   |
|{named_entity, 30, 31, O, {word -> to, confidence -> 0.7063}, []}         |
+--------------------------------------------------------------------------+

# After the converter is used:
result.selectExpr("explode(ner_converter_result)").show(5, False)
+-----------------------------------------------------------------------------------+
|col                                                                                |
+-----------------------------------------------------------------------------------+
|{chunk, 5, 15, 63-year-old, {entity -> Age, sentence -> 0, chunk -> 0}, []}        |
|{chunk, 17, 19, man, {entity -> Gender, sentence -> 0, chunk -> 1}, []}            |
|{chunk, 64, 72, recurrent, {entity -> Modifier, sentence -> 0, chunk -> 2}, []}    |
|{chunk, 98, 107, cellulitis, {entity -> Diagnosis, sentence -> 0, chunk -> 3}, []} |
|{chunk, 110, 119, pneumonias, {entity -> Diagnosis, sentence -> 0, chunk -> 4}, []}|
+-----------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
// The output of a [[MedicalNerModel]] follows the Annotator schema and looks like this after the transformation.
//
// result.selectExpr("explode(ner_result)").show(5, false)
// +--------------------------------------------------------------------------+
// |col                                                                       |
// +--------------------------------------------------------------------------+
// |{named_entity, 3, 3, O, {word -> A, confidence -> 0.994}, []}             |
// |{named_entity, 5, 15, B-Age, {word -> 63-year-old, confidence -> 1.0}, []}|
// |{named_entity, 17, 19, B-Gender, {word -> man, confidence -> 0.9858}, []} |
// |{named_entity, 21, 28, O, {word -> presents, confidence -> 0.9952}, []}   |
// |{named_entity, 30, 31, O, {word -> to, confidence -> 0.7063}, []}         |
// +--------------------------------------------------------------------------+
//
// After the converter is used:
//
// result.selectExpr("explode(ner_converter_result)").show(5, false)
// +-----------------------------------------------------------------------------------+
// |col                                                                                |
// +-----------------------------------------------------------------------------------+
// |{chunk, 5, 15, 63-year-old, {entity -> Age, sentence -> 0, chunk -> 0}, []}        |
// |{chunk, 17, 19, man, {entity -> Gender, sentence -> 0, chunk -> 1}, []}            |
// |{chunk, 64, 72, recurrent, {entity -> Modifier, sentence -> 0, chunk -> 2}, []}    |
// |{chunk, 98, 107, cellulitis, {entity -> Diagnosis, sentence -> 0, chunk -> 3}, []} |
// |{chunk, 110, 119, pneumonias, {entity -> Diagnosis, sentence -> 0, chunk -> 4}, []}|
// +-----------------------------------------------------------------------------------+
//
{%- endcapture -%}

{%- capture api_link -%}
[NerConverterInternal](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/NerConverterInternal)
{%- endcapture -%}

{% include templates/licensed_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link%}