---
layout: model
title: Named Entity Recognition - BERT Large (OntoNotes)
author: John Snow Labs
name: onto_bert_large_cased
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

This model uses the pretrained `bert_large_cased` embeddings model from the `BertEmbeddings` annotator as an input.

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_bert_large_cased_en_2.7.0_2.4_1607198127113.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/onto_bert_large_cased_en_2.7.0_2.4_1607198127113.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertEmbeddings.pretrained("bert_large_cased", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
ner_onto = NerDLModel.pretrained("onto_bert_large_cased", "en") \
.setInputCols(["document", "token", "embeddings"]) \
.setOutputCol("ner")
...        
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, ner_onto, ner_converter])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([['']]).toDF('text'))

result = pipeline_model.transform(spark.createDataFrame([["William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect, while also being the largest individual shareholder until May 2014. He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico; it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect. During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000. He gradually transferred his duties to Ray Ozzie and Craig Mundie. He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella."]], ["text"]))
```

```scala
...
val embeddings = BertEmbeddings.pretrained("bert_large_cased", "en")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")
val ner_onto = NerDLModel.pretrained("onto_bert_large_cased", "en")
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
ner_df = nlu.load('en.ner.onto.bert.cased_large').predict(text, output_level='chunk')
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
|one                    |CARDINAL |
|the 1970s              |DATE     |
|1980s                  |DATE     |
|Seattle                |GPE      |
|Washington             |GPE      |
|Gates                  |PERSON   |
|Microsoft              |PERSON   |
|Paul Allen             |PERSON   |
|1975                   |DATE     |
|Albuquerque            |GPE      |
|New Mexico             |GPE      |
|Gates                  |ORG      |
|January 2000           |DATE     |
+-----------------------+---------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_bert_large_cased|
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

prec: 0.8947816, rec: 0.9059915, f1: 0.90035164

CoNLL Eval (test dataset):

processed 152728 tokens with 11257 phrases; found: 11351 phrases; correct: 10044.
accuracy:  98.02%; 10044 11257 11351 precision:  88.49%; recall:  89.22%; FB1:  88.85
CARDINAL:  793  935  953 precision:  83.21%; recall:  84.81%; FB1:  84.00  953
DATE:  1420  1602  1697 precision:  83.68%; recall:  88.64%; FB1:  86.09  1697
EVENT:   37   63   63 precision:  58.73%; recall:  58.73%; FB1:  58.73  63
FAC:   98  135  152 precision:  64.47%; recall:  72.59%; FB1:  68.29  152
GPE:  2128  2240  2218 precision:  95.94%; recall:  95.00%; FB1:  95.47  2218
LANGUAGE:   10   22   13 precision:  76.92%; recall:  45.45%; FB1:  57.14  13
LAW:   21   40   30 precision:  70.00%; recall:  52.50%; FB1:  60.00  30
LOC:  133  179  166 precision:  80.12%; recall:  74.30%; FB1:  77.10  166
MONEY:  279  314  317 precision:  88.01%; recall:  88.85%; FB1:  88.43  317
NORP:  796  841  840 precision:  94.76%; recall:  94.65%; FB1:  94.71  840
ORDINAL:  180  195  219 precision:  82.19%; recall:  92.31%; FB1:  86.96  219
ORG:  1620  1795  1873 precision:  86.49%; recall:  90.25%; FB1:  88.33  1873
PERCENT:  309  349  342 precision:  90.35%; recall:  88.54%; FB1:  89.44  342
PERSON:  1862  1988  1970 precision:  94.52%; recall:  93.66%; FB1:  94.09  1970
PRODUCT:   51   76   68 precision:  75.00%; recall:  67.11%; FB1:  70.83  68
QUANTITY:   81  105   99 precision:  81.82%; recall:  77.14%; FB1:  79.41  99
TIME:  116  212  179 precision:  64.80%; recall:  54.72%; FB1:  59.34  179
WORK_OF_ART:  110  166  152 precision:  72.37%; recall:  66.27%; FB1:  69.18  152

CoNLL Eval (development dataset):

processed 147724 tokens with 11066 phrases; found: 11259 phrases; correct: 10326.
accuracy:  98.73%; 10326 11066 11259 precision:  91.71%; recall:  93.31%; FB1:  92.51
CARDINAL:  869  938  1001 precision:  86.81%; recall:  92.64%; FB1:  89.63  1001
DATE:  1374  1507  1571 precision:  87.46%; recall:  91.17%; FB1:  89.28  1571
EVENT:  112  143  138 precision:  81.16%; recall:  78.32%; FB1:  79.72  138
FAC:  105  115  140 precision:  75.00%; recall:  91.30%; FB1:  82.35  140
GPE:  2213  2268  2293 precision:  96.51%; recall:  97.57%; FB1:  97.04  2293
LANGUAGE:   27   33   27 precision: 100.00%; recall:  81.82%; FB1:  90.00  27
LAW:   33   40   39 precision:  84.62%; recall:  82.50%; FB1:  83.54  39
LOC:  163  204  178 precision:  91.57%; recall:  79.90%; FB1:  85.34  178
MONEY:  257  274  275 precision:  93.45%; recall:  93.80%; FB1:  93.62  275
NORP:  806  847  841 precision:  95.84%; recall:  95.16%; FB1:  95.50  841
ORDINAL:  212  232  241 precision:  87.97%; recall:  91.38%; FB1:  89.64  241
ORG:  1618  1740  1771 precision:  91.36%; recall:  92.99%; FB1:  92.17  1771
PERCENT:  164  177  174 precision:  94.25%; recall:  92.66%; FB1:  93.45  174
PERSON:  1961  2020  2064 precision:  95.01%; recall:  97.08%; FB1:  96.03  2064
PRODUCT:   64   72   73 precision:  87.67%; recall:  88.89%; FB1:  88.28  73
QUANTITY:   83  100   96 precision:  86.46%; recall:  83.00%; FB1:  84.69  96
TIME:  165  214  203 precision:  81.28%; recall:  77.10%; FB1:  79.14  203
WORK_OF_ART:  100  142  134 precision:  74.63%; recall:  70.42%; FB1:  72.46  134

```
