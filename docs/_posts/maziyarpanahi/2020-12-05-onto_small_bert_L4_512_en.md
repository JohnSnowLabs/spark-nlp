---
layout: model
title: Named Entity Recognition - BERT Small (OntoNotes)
author: John Snow Labs
name: onto_small_bert_L4_512
date: 2020-12-05
task: Named Entity Recognition
language: en
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [ner, open_source, en]
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Onto is a Named Entity Recognition (or NER) model trained on OntoNotes 5.0. It can extract up to 18 entities such as people, places, organizations, money, time, date, etc.

This model uses the pretrained `small_bert_L4_512` embeddings model from the `BertEmbeddings` annotator as an input.

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_small_bert_L4_512_en_2.7.0_2.4_1607199400149.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/onto_small_bert_L4_512_en_2.7.0_2.4_1607199400149.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
ner_onto = NerDLModel.pretrained("onto_small_bert_L4_512", "en") \
.setInputCols(["document", "token", "embeddings"]) \
.setOutputCol("ner")
...        
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, ner_onto, ner_converter])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([['']]).toDF('text'))

result = pipeline_model.transform(spark.createDataFrame([["William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect, while also being the largest individual shareholder until May 2014. He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico; it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect. During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000. He gradually transferred his duties to Ray Ozzie and Craig Mundie. He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella."]], ["text"]))
```

```scala
...
val ner_onto = NerDLModel.pretrained("onto_small_bert_L4_512", "en")
.setInputCols(Array("document", "token", "embeddings"))
.setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings, ner_onto, ner_converter))

val data = Seq("William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect, while also being the largest individual shareholder until May 2014. He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico; it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect. During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000. He gradually transferred his duties to Ray Ozzie and Craig Mundie. He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect, while also being the largest individual shareholder until May 2014. He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico; it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect. During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000. He gradually transferred his duties to Ray Ozzie and Craig Mundie. He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella."""]
ner_df = nlu.load('en.ner.onto.bert.small_l4_512').predict(text, output_level='chunk')
ner_df[["entities", "entities_class"]]
```

</div>

{:.h2_title}
## Results

```bash

+-----------------------+---------+
|chunk                  |ner_label|
+-----------------------+---------+
|William Henry Gates III|PERSON   |
|October 28, 1955       |DATE     |
|American               |NORP     |
|Microsoft Corporation  |ORG      |
|Microsoft              |ORG      |
|Gates                  |PERSON   |
|May 2014               |DATE     |
|the 1970s              |DATE     |
|1980s                  |DATE     |
|Seattle                |GPE      |
|Washington             |GPE      |
|Gates                  |PERSON   |
|Paul Allen             |PERSON   |
|1975                   |DATE     |
|Albuquerque            |GPE      |
|New Mexico             |GPE      |
|Gates                  |PERSON   |
|January 2000           |DATE     |
|the late 1990s         |DATE     |
|Gates                  |PERSON   |
+-----------------------+---------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_small_bert_L4_512|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

The model is trained based on data from [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)

## Benchmarking

```bash
Micro-average:

prec: 0.8697573, rec: 0.8567398, f1: 0.8631994

CoNLL Eval:

processed 152728 tokens with 11257 phrases; found: 11219 phrases; correct: 9557.
accuracy:  97.17%; 9557 11257 11219 precision:  85.19%; recall:  84.90%; FB1:  85.04
CARDINAL:  804  935  958 precision:  83.92%; recall:  85.99%; FB1:  84.94  958
DATE:  1412  1602  1726 precision:  81.81%; recall:  88.14%; FB1:  84.86  1726
EVENT:   20   63   46 precision:  43.48%; recall:  31.75%; FB1:  36.70  46
FAC:   78  135  122 precision:  63.93%; recall:  57.78%; FB1:  60.70  122
GPE:  2066  2240  2185 precision:  94.55%; recall:  92.23%; FB1:  93.38  2185
LANGUAGE:   10   22   11 precision:  90.91%; recall:  45.45%; FB1:  60.61  11
LAW:   12   40   18 precision:  66.67%; recall:  30.00%; FB1:  41.38  18
LOC:  114  179  168 precision:  67.86%; recall:  63.69%; FB1:  65.71  168
MONEY:  273  314  320 precision:  85.31%; recall:  86.94%; FB1:  86.12  320
NORP:  779  841  873 precision:  89.23%; recall:  92.63%; FB1:  90.90  873
ORDINAL:  174  195  226 precision:  76.99%; recall:  89.23%; FB1:  82.66  226
ORG:  1381  1795  1691 precision:  81.67%; recall:  76.94%; FB1:  79.23  1691
PERCENT:  311  349  349 precision:  89.11%; recall:  89.11%; FB1:  89.11  349
PERSON:  1827  1988  2046 precision:  89.30%; recall:  91.90%; FB1:  90.58  2046
PRODUCT:   32   76   51 precision:  62.75%; recall:  42.11%; FB1:  50.39  51
QUANTITY:   80  105  105 precision:  76.19%; recall:  76.19%; FB1:  76.19  105
TIME:  124  212  219 precision:  56.62%; recall:  58.49%; FB1:  57.54  219
WORK_OF_ART:   60  166  105 precision:  57.14%; recall:  36.14%; FB1:  44.28  105
```