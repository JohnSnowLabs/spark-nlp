{%- capture title -%}
Annotation tool json reader.
{%- endcapture -%}

{%- capture description -%}
The annotation tool json reader is a reader that generate a assertion train set from the json from annotations labs exports.

{%- endcapture -%}


{%- capture file_format -%}
[
  {
    "completions": [
      {
        "created_ago": "2020-05-18T20:48:18.117Z",
        "created_username": "admin",
        "id": 3001,
        "lead_time": 19.255,
        "result": [
          {
            "from_name": "ner",
            "id": "o752YyB2g9",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 12,
              "labels": [
                "AsPresent"
              ],
              "start": 3,
              "text": "have faith"
            }
          },
          {
            "from_name": "ner",
            "id": "wf2U3o7I6T",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 24,
              "labels": [
                "AsPresent"
              ],
              "start": 16,
              "text": " to trust"
            }
          },
          {
            "from_name": "ner",
            "id": "Q3BkU5eZNx",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 40,
              "labels": [
                "AsPresent"
              ],
              "start": 35,
              "text": "to the"
            }
          }
        ]
      }
    ],
    "created_at": "2020-05-18 20:47:53",
    "created_by": "andres.fernandez",
    "data": {
      "text": "To have faith is to trust yourself to the water"
    },
    "id": 3
  },
  {
    "completions": [
      {
        "created_ago": "2020-05-17T17:52:41.563Z",
        "created_username": "andres.fernandez",
        "id": 1,
        "lead_time": 31.449,
        "result": [
          {
            "from_name": "ner",
            "id": "IQjoZJNKEv",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 12,
              "labels": [
                "Disease"
              ],
              "start": 3,
              "text": "have faith"
            }
          },
          {
            "from_name": "ner",
            "id": "tHsbn4oYy5",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 46,
              "labels": [
                "Treatment"
              ],
              "start": 42,
              "text": "water"
            }
          },
          {
            "from_name": "ner",
            "id": "IJHkc9bxJ-",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 12,
              "labels": [
                "AsPresent"
              ],
              "start": 0,
              "text": "To have faith"
            }
          }
        ]
      }
    ],
    "created_at": "2020-05-17 17:52:02",
    "created_by": "andres.fernandez",
    "data": {
      "text": "To have faith is to trust yourself to the water"
    },
    "id": 0
  },
  {
    "completions": [
      {
        "created_ago": "2020-05-17T17:57:19.402Z",
        "created_username": "andres.fernandez",
        "id": 1001,
        "lead_time": 15.454,
        "result": [
          {
            "from_name": "ner",
            "id": "j_lT0zwtrJ",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 46,
              "labels": [
                "Disease"
              ],
              "start": 20,
              "text": "trust yourself to the water"
            }
          },
          {
            "from_name": "ner",
            "id": "e1FuGWu7EQ",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 33,
              "labels": [
                "AsPresent"
              ],
              "start": 19,
              "text": " trust yourself"
            }
          },
          {
            "from_name": "ner",
            "id": "q0MCSM9SXz",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 12,
              "labels": [
                "Treatment"
              ],
              "start": 0,
              "text": "To have faith"
            }
          },
          {
            "from_name": "ner",
            "id": "9R7dvPphPX",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 12,
              "labels": [
                "AsPresent"
              ],
              "start": 0,
              "text": "To have faith"
            }
          }
        ]
      }
    ],
    "created_at": "2020-05-17 17:52:54",
    "created_by": "andres.fernandez",
    "data": {
      "text": "To have faith is to trust yourself to the water"
    },
    "id": 1,
    "predictions": []
  }
]

{%- endcapture -%}


{%- capture constructor -%}
- **assertion_labels**:  The assertions labels are used for the training dataset creation.
- **excluded_labels**:  The assertions labels that are excluded for the training dataset creation.
- **split_chars**:  The split chars that are used in the default tokenizer.
- **context_chars**: The context chars that are used in the default tokenizer.
- **SDDLPath**: The context chars that are used in the default tokenizer.
{%- endcapture -%}

{%- capture read_dataset_params -%}
- **spark**: Initiated Spark Session with Spark NLP
- **path**: Path to the resource
{%- endcapture -%}

{%- capture python_example -%}

from sparknlp_jsl.training import AnnotationToolJsonReader
assertion_labels = ["AsPresent","Absent"]
excluded_labels = ["Treatment"]
split_chars = [" ", "\\-"]
context_chars = [".", ",", ";", ":", "!", "?", "*", "-", "(", ")", "\"", "'","+","%","'"]
SDDLPath = ""
rdr = AnnotationToolJsonReader(assertion_labels = assertion_labels, excluded_labels = excluded_labels, split_chars = split_chars, context_chars = context_chars,SDDLPath=SDDLPath)
path = "src/test/resources/anc-pos-corpus-small/test-training.txt"
df = rdr.readDataset(spark, json_path)
assertion_df = rdr.generateAssertionTrainSet(df)
assertion_df.show()

+--------------------+--------------+---------+-----+---+
|                text|        target|    label|start|end|
+--------------------+--------------+---------+-----+---+
|To have faith is ...| To have faith|AsPresent|    0|  2|
|To have faith is ...|    have faith|AsPresent|    1|  2|
|To have faith is ...|      to trust|AsPresent|    4|  5|
|To have faith is ...|        to the|AsPresent|    7|  8|
|To have faith is ...|      yourself|AsPresent|    6|  6|
|To have faith is ...| To have faith|AsPresent|    0|  2|
|To have faith is ...|trust yourself|AsPresent|    5|  6|
+--------------------+--------------+---------+-----+---+


{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.training.POS

val filename = "src/test/resources/json_import.json"

val reader = new AnnotationToolJsonReader(assertionLabels=List("AsPresent","Absent").asJava, splitChars=List(" ", "\\-").asJava, excludedLabels = List("Treatment").asJava)
val df = reader.readDataset(ResourceHelper.spark, filename)
val assertionDf = reader.generateAssertionTrainSet(df)
assertionDf.show()

+--------------------+--------------+---------+-----+---+
|                text|        target|    label|start|end|
+--------------------+--------------+---------+-----+---+
|To have faith is ...| To have faith|AsPresent|    0|  2|
|To have faith is ...|    have faith|AsPresent|    1|  2|
|To have faith is ...|      to trust|AsPresent|    4|  5|
|To have faith is ...|        to the|AsPresent|    7|  8|
|To have faith is ...|      yourself|AsPresent|    6|  6|
|To have faith is ...| To have faith|AsPresent|    0|  2|
|To have faith is ...|trust yourself|AsPresent|    5|  6|
+--------------------+--------------+---------+-----+---+

{%- endcapture -%}

{%- capture api_link -%}

{%- endcapture -%}

{%- capture python_api_link -%}

{%- endcapture -%}

{%- capture api_link -%}

[AnnotationToolJsonReader](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/training/AnnotationToolJsonReader.html)

{%- endcapture -%}

{% include templates/licensed_training_dataset_entry.md
title=title
description=description
file_format=file_format
constructor=constructor
read_dataset_params=read_dataset_params
python_example=python_example
scala_example=scala_example
api_link=api_link
%}
