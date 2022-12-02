{%- capture title -%}
CoNLL Dataset
{%- endcapture -%}

{%- capture description -%}
In order to train a Named Entity Recognition DL annotator, we need to get CoNLL format data as a spark dataframe. There is a component that does this for us: it reads a plain text file and transforms it to a spark dataset.

The dataset should be in the format of [CoNLL 2003](https://www.clips.uantwerpen.be/conll2003/ner/) and needs to be specified with `readDataset()`, which will create a dataframe with the data.
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
- **documentCol:** Name of the DocumentAssembler column, by default ‘document’
- **sentenceCol:** Name of the SentenceDetector column, by default ‘sentence’
- **tokenCol:** Name of the Tokenizer column, by default ‘token’
- **posCol:** Name of the part-of-speech tag column, by default ‘pos’
- **conllLabelIndex:** Index of the label column in the dataset, by default 3
- **conllPosIndex:** Index of the POS tags in the dataset, by default 1
- **textCol:** Index of the text column in the dataset, by default ‘text’
- **labelCol:** Name of the label column, by default ‘label’
- **explodeSentences:** Whether to explode sentences to separate rows, by default True
- **delimiter:** Delimiter used to separate columns inside CoNLL file
{%- endcapture -%}

{%- capture read_dataset_params -%}
- **spark**: Initiated Spark Session with Spark NLP
- **path**: Path to the resource
- **read_as**: How to read the resource, by default ReadAs.TEXT
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.training import CoNLL
trainingData = CoNLL().readDataset(spark, "src/test/resources/conll2003/eng.train")
trainingData.selectExpr(
    "text",
    "token.result as tokens",
    "pos.result as pos",
    "label.result as label"
).show(3, False)
+------------------------------------------------+----------------------------------------------------------+-------------------------------------+-----------------------------------------+
|text                                            |tokens                                                    |pos                                  |label                                    |
+------------------------------------------------+----------------------------------------------------------+-------------------------------------+-----------------------------------------+
|EU rejects German call to boycott British lamb .|[EU, rejects, German, call, to, boycott, British, lamb, .]|[NNP, VBZ, JJ, NN, TO, VB, JJ, NN, .]|[B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]|
|Peter Blackburn                                 |[Peter, Blackburn]                                        |[NNP, NNP]                           |[B-PER, I-PER]                           |
|BRUSSELS 1996-08-22                             |[BRUSSELS, 1996-08-22]                                    |[NNP, CD]                            |[B-LOC, O]                               |
+------------------------------------------------+----------------------------------------------------------+-------------------------------------+-----------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
val trainingData = CoNLL().readDataset(spark, "src/test/resources/conll2003/eng.train")
trainingData.selectExpr("text", "token.result as tokens", "pos.result as pos", "label.result as label")
  .show(3, false)
+------------------------------------------------+----------------------------------------------------------+-------------------------------------+-----------------------------------------+
|text                                            |tokens                                                    |pos                                  |label                                    |
+------------------------------------------------+----------------------------------------------------------+-------------------------------------+-----------------------------------------+
|EU rejects German call to boycott British lamb .|[EU, rejects, German, call, to, boycott, British, lamb, .]|[NNP, VBZ, JJ, NN, TO, VB, JJ, NN, .]|[B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]|
|Peter Blackburn                                 |[Peter, Blackburn]                                        |[NNP, NNP]                           |[B-PER, I-PER]                           |
|BRUSSELS 1996-08-22                             |[BRUSSELS, 1996-08-22]                                    |[NNP, CD]                            |[B-LOC, O]                               |
+------------------------------------------------+----------------------------------------------------------+-------------------------------------+-----------------------------------------+
{%- endcapture -%}

{%- capture api_link -%}
[CoNLL](/api/com/johnsnowlabs/nlp/training/CoNLL.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[CoNLL](/api/python/reference/autosummary/python/sparknlp/training/conll/index.html#sparknlp.training.conll.CoNLL)
{%- endcapture -%}

{%- capture source_link -%}
[CoNLL.scala](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/training/CoNLL.scala)
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