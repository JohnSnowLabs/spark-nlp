{%- capture title -%}
CoNLL-U Dataset
{%- endcapture -%}

{%- capture description -%}
In order to train a DependencyParserApproach annotator, we need to get CoNLL-U format data as a spark dataframe. There is a component that does this for us: it reads a plain text file and transforms it to a spark dataset.

The dataset should be in the format of [CoNLL-U](https://universaldependencies.org/format.html) and needs to be specified with `readDataset()`, which will create a dataframe with the data.
{%- endcapture -%}

{%- capture file_format -%}
-DOCSTART- -X- -X- O

EU NNP B-NP B-ORG
rejects VBZ B-VP O
German JJ B-NP B-MISC
call NN I-NP O
to TO B-VP O
boycott VB I-VP O
British JJ B-NP B-MISC
lamb NN I-NP O
. . O O
{%- endcapture -%}

{%- capture constructor -%}
- **explodeSentences**: Whether to explode each sentence to a separate row
{%- endcapture -%}

{%- capture read_dataset_params -%}
- **spark**: Initiated Spark Session with Spark NLP
- **path**: Path to the resource
- **read_as**: How to read the resource, by default ReadAs.TEXT
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.training import CoNLLU
conlluFile = "src/test/resources/conllu/en.test.conllu"
conllDataSet = CoNLLU(False).readDataset(spark, conlluFile)
conllDataSet.selectExpr(
    "text",
    "form.result as form",
    "upos.result as upos",
    "xpos.result as xpos",
    "lemma.result as lemma"
).show(1, False)
+---------------------------------------+----------------------------------------------+---------------------------------------------+------------------------------+--------------------------------------------+
|text                                   |form                                          |upos                                         |xpos                          |lemma                                       |
+---------------------------------------+----------------------------------------------+---------------------------------------------+------------------------------+--------------------------------------------+
|What if Google Morphed Into GoogleOS?  |[What, if, Google, Morphed, Into, GoogleOS, ?]|[PRON, SCONJ, PROPN, VERB, ADP, PROPN, PUNCT]|[WP, IN, NNP, VBD, IN, NNP, .]|[what, if, Google, morph, into, GoogleOS, ?]|
+---------------------------------------+----------------------------------------------+---------------------------------------------+------------------------------+--------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.training.CoNLLU

val conlluFile = "src/test/resources/conllu/en.test.conllu"
val conllDataSet = CoNLLU(false).readDataset(ResourceHelper.spark, conlluFile)
conllDataSet.selectExpr("text", "form.result as form", "upos.result as upos", "xpos.result as xpos", "lemma.result as lemma")
  .show(1, false)
+---------------------------------------+----------------------------------------------+---------------------------------------------+------------------------------+--------------------------------------------+
|text                                   |form                                          |upos                                         |xpos                          |lemma                                       |
+---------------------------------------+----------------------------------------------+---------------------------------------------+------------------------------+--------------------------------------------+
|What if Google Morphed Into GoogleOS?  |[What, if, Google, Morphed, Into, GoogleOS, ?]|[PRON, SCONJ, PROPN, VERB, ADP, PROPN, PUNCT]|[WP, IN, NNP, VBD, IN, NNP, .]|[what, if, Google, morph, into, GoogleOS, ?]|
+---------------------------------------+----------------------------------------------+---------------------------------------------+------------------------------+--------------------------------------------+
{%- endcapture -%}

{%- capture api_link -%}
[CoNLLU](/api/com/johnsnowlabs/nlp/training/CoNLLU.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[CoNLLU](/api/python/reference/autosummary/sparknlp.training.CoNLLU.html)
{%- endcapture -%}

{%- capture source_link -%}
[CoNLLU.scala](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/training/CoNLLU.scala)
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