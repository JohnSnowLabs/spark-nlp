{%- capture title -%}
CoNLL-U Dataset
{%- endcapture -%}

{%- capture description -%}
In order to train a DependencyParserApproach annotator, we need to get CoNLL-U format data as a spark dataframe. There is a component that does this for us: it reads a plain text file and transforms it to a spark dataset.

The dataset should be in the format of [CoNLL-U](https://universaldependencies.org/format.html) and needs to be specified with `readDataset()`, which will create a dataframe with the data.
{%- endcapture -%}

{%- capture file_format -%}
# sent_id = 1
# text = They buy and sell books.
1   They     they    PRON    PRP    Case=Nom|Number=Plur               2   nsubj   2:nsubj|4:nsubj   _
2   buy      buy     VERB    VBP    Number=Plur|Person=3|Tense=Pres    0   root    0:root            _
3   and      and     CONJ    CC     _                                  4   cc      4:cc              _
4   sell     sell    VERB    VBP    Number=Plur|Person=3|Tense=Pres    2   conj    0:root|2:conj     _
5   books    book    NOUN    NNS    Number=Plur                        2   obj     2:obj|4:obj       SpaceAfter=No
6   .        .       PUNCT   .      _                                  2   punct   2:punct           _
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
[CoNLLU](/api/python/reference/autosummary/python/sparknlp/training/conllu/index.html#sparknlp.training.conllu.CoNLLU)
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