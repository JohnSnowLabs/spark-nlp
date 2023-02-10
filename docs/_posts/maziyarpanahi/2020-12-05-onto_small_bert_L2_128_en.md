---
layout: model
title: Named Entity Recognition - BERT Tiny (OntoNotes)
author: John Snow Labs
name: onto_small_bert_L2_128
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

This model uses the pretrained `small_bert_L2_128` embeddings model from the `BertEmbeddings` annotator as an input.

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_small_bert_L2_128_en_2.7.0_2.4_1607198998042.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/onto_small_bert_L2_128_en_2.7.0_2.4_1607198998042.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
ner_onto = NerDLModel.pretrained("onto_small_bert_L2_128", "en") \
.setInputCols(["document", "token", "embeddings"]) \
.setOutputCol("ner")
...        
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, ner_onto, ner_converter])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([['']]).toDF('text'))

result = pipeline_model.transform(spark.createDataFrame([["William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect, while also being the largest individual shareholder until May 2014. He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico; it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect. During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000. He gradually transferred his duties to Ray Ozzie and Craig Mundie. He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella."]], ["text"]))
```

```scala
...
val ner_onto = NerDLModel.pretrained("onto_small_bert_L2_128", "en")
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
ner_df = nlu.load('en.ner.onto.bert.small_l2_128').predict(text, output_level='chunk')
ner_df[["entities", "entities_class"]]
```

</div>

{:.h2_title}
## Results

```bash

+----------------------------+---------+
|chunk                       |ner_label|
+----------------------------+---------+
|William Henry Gates III     |PERSON   |
|October 28, 1955            |DATE     |
|American                    |NORP     |
|Microsoft Corporation       |ORG      |
|Microsoft                   |ORG      |
|Gates                       |PERSON   |
|May 2014                    |DATE     |
|the microcomputer revolution|EVENT    |
|1970s                       |DATE     |
|1980s                       |DATE     |
|Seattle                     |GPE      |
|Washington                  |GPE      |
|Paul Allen                  |PERSON   |
|1975                        |DATE     |
|Albuquerque                 |GPE      |
|New Mexico                  |GPE      |
|Gates                       |PERSON   |
|January 2000                |DATE     |
|the late 1990s              |DATE     |
|Gates                       |PERSON   |
+----------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_small_bert_L2_128|
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

prec: 0.86477494, rec: 0.8204466, f1: 0.8420278

CoNLL Eval:

processed 152728 tokens with 11257 phrases; found: 10772 phrases; correct: 9153.
accuracy:  96.73%; 9153 11257 10772 precision:  84.97%; recall:  81.31%; FB1:  83.10
CARDINAL:  733  935  890 precision:  82.36%; recall:  78.40%; FB1:  80.33  890
DATE:  1278  1602  1494 precision:  85.54%; recall:  79.78%; FB1:  82.56  1494
EVENT:   22   63   45 precision:  48.89%; recall:  34.92%; FB1:  40.74  45
FAC:   67  135  114 precision:  58.77%; recall:  49.63%; FB1:  53.82  114
GPE:  2044  2240  2201 precision:  92.87%; recall:  91.25%; FB1:  92.05  2201
LANGUAGE:    8   22   14 precision:  57.14%; recall:  36.36%; FB1:  44.44  14
LAW:   12   40   15 precision:  80.00%; recall:  30.00%; FB1:  43.64  15
LOC:  104  179  155 precision:  67.10%; recall:  58.10%; FB1:  62.28  155
MONEY:  265  314  316 precision:  83.86%; recall:  84.39%; FB1:  84.13  316
NORP:  775  841  886 precision:  87.47%; recall:  92.15%; FB1:  89.75  886
ORDINAL:  180  195  239 precision:  75.31%; recall:  92.31%; FB1:  82.95  239
ORG:  1280  1795  1548 precision:  82.69%; recall:  71.31%; FB1:  76.58  1548
PERCENT:  308  349  350 precision:  88.00%; recall:  88.25%; FB1:  88.13  350
PERSON:  1784  1988  2032 precision:  87.80%; recall:  89.74%; FB1:  88.76  2032
PRODUCT:   33   76   49 precision:  67.35%; recall:  43.42%; FB1:  52.80  49
QUANTITY:   83  105  112 precision:  74.11%; recall:  79.05%; FB1:  76.50  112
TIME:  124  212  205 precision:  60.49%; recall:  58.49%; FB1:  59.47  205
WORK_OF_ART:   53  166  107 precision:  49.53%; recall:  31.93%; FB1:  38.83  107
```