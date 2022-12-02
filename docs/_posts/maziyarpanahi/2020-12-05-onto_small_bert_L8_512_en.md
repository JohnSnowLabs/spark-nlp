---
layout: model
title: Named Entity Recognition - BERT Medium (OntoNotes)
author: John Snow Labs
name: onto_small_bert_L8_512
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

This model uses the pretrained `small_bert_L8_512` embeddings model from the `BertEmbeddings` annotator as an input.

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_small_bert_L8_512_en_2.7.0_2.4_1607199531477.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
ner_onto = NerDLModel.pretrained("onto_small_bert_L8_512", "en") \
.setInputCols(["document", "token", "embeddings"]) \
.setOutputCol("ner")
...        
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, ner_onto, ner_converter])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([['']]).toDF('text'))

result = pipeline_model.transform(spark.createDataFrame([["William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect, while also being the largest individual shareholder until May 2014. He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico; it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect. During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000. He gradually transferred his duties to Ray Ozzie and Craig Mundie. He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella."]], ["text"]))
```

```scala
...
val ner_onto = NerDLModel.pretrained("onto_small_bert_L8_512", "en")
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
ner_df = nlu.load('en.ner.onto.bert.small_l8_512').predict(text, output_level='chunk')
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
|the 1970s and 1980s    |DATE     |
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
|June 2006              |DATE     |
+-----------------------+---------+

```
{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_small_bert_L8_512|
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

prec: 0.8849518, rec: 0.85147995, f1: 0.8678933

CoNLL Eval:

processed 152728 tokens with 11257 phrases; found: 11073 phrases; correct: 9556.
accuracy:  97.26%; 9556 11257 11073 precision:  86.30%; recall:  84.89%; FB1:  85.59
CARDINAL:  798  935  929 precision:  85.90%; recall:  85.35%; FB1:  85.62  929
DATE:  1410  1602  1654 precision:  85.25%; recall:  88.01%; FB1:  86.61  1654
EVENT:   23   63   44 precision:  52.27%; recall:  36.51%; FB1:  42.99  44
FAC:   79  135  121 precision:  65.29%; recall:  58.52%; FB1:  61.72  121
GPE:  2097  2240  2244 precision:  93.45%; recall:  93.62%; FB1:  93.53  2244
LANGUAGE:    9   22   11 precision:  81.82%; recall:  40.91%; FB1:  54.55  11
LAW:   14   40   20 precision:  70.00%; recall:  35.00%; FB1:  46.67  20
LOC:  111  179  152 precision:  73.03%; recall:  62.01%; FB1:  67.07  152
MONEY:  282  314  320 precision:  88.12%; recall:  89.81%; FB1:  88.96  320
NORP:  755  841  889 precision:  84.93%; recall:  89.77%; FB1:  87.28  889
ORDINAL:  169  195  201 precision:  84.08%; recall:  86.67%; FB1:  85.35  201
ORG:  1368  1795  1624 precision:  84.24%; recall:  76.21%; FB1:  80.02  1624
PERCENT:  309  349  351 precision:  88.03%; recall:  88.54%; FB1:  88.29  351
PERSON:  1816  1988  2037 precision:  89.15%; recall:  91.35%; FB1:  90.24  2037
PRODUCT:   42   76   67 precision:  62.69%; recall:  55.26%; FB1:  58.74  67
QUANTITY:   85  105  108 precision:  78.70%; recall:  80.95%; FB1:  79.81  108
TIME:  137  212  222 precision:  61.71%; recall:  64.62%; FB1:  63.13  222
WORK_OF_ART:   52  166   79 precision:  65.82%; recall:  31.33%; FB1:  42.45  79
```