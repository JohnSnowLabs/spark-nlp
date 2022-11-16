---
layout: model
title: Named Entity Recognition - ELECTRA Large (OntoNotes)
author: John Snow Labs
name: onto_electra_large_uncased
date: 2020-12-05
task: Named Entity Recognition
language: en
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [ner, en, open_source]
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Onto is a Named Entity Recognition (or NER) model trained on OntoNotes 5.0. It can extract up to 18 entities such as people, places, organizations, money, time, date, etc.

This model uses the pretrained `electra_large_uncased` embeddings model from the `BertEmbeddings` annotator as an input.

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_electra_large_uncased_en_2.7.0_2.4_1607198670231.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = BertEmbeddings.pretrained("electra_large_uncased", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
ner_onto = NerDLModel.pretrained("onto_electra_large_uncased", "en") \
.setInputCols(["document", "token", "embeddings"]) \
.setOutputCol("ner")
...        
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, ner_onto, ner_converter])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([['']]).toDF('text'))

result = pipeline_model.transform(spark.createDataFrame([["William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect, while also being the largest individual shareholder until May 2014. He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico; it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect. During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000. He gradually transferred his duties to Ray Ozzie and Craig Mundie. He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella."]], ["text"]))
```

```scala
...
val embeddings = BertEmbeddings.pretrained("electra_large_uncased", "en")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")
val ner_onto = NerDLModel.pretrained("onto_electra_large_uncased", "en")
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
ner_df = nlu.load('en.ner.onto.electra.uncased_large').predict(text, output_level='chunk')
ner_df[["entities", "entities_class"]]
```

</div>

{:.h2_title}
## Results

```bash

+---------------------+---------+
|chunk                |ner_label|
+---------------------+---------+
|William Henry Gates  |PERSON   |
|October 28, 1955     |DATE     |
|American             |NORP     |
|Microsoft Corporation|ORG      |
|Microsoft            |ORG      |
|Gates                |PERSON   |
|CEO                  |ORG      |
|May 2014             |DATE     |
|one                  |CARDINAL |
|the 1970s and 1980s  |DATE     |
|Seattle              |GPE      |
|Washington           |GPE      |
|Gates                |PERSON   |
|Microsoft            |FAC      |
|Paul Allen           |PERSON   |
|1975                 |DATE     |
|Albuquerque          |GPE      |
|New Mexico           |GPE      |
|Gates                |PERSON   |
|January 2000         |DATE     |
+---------------------+---------+

```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_electra_large_uncased|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

The model is trained based on data from OntoNotes 5.0 [https://catalog.ldc.upenn.edu/LDC2013T19](https://catalog.ldc.upenn.edu/LDC2013T19)

## Benchmarking

```bash
Micro-average:

prec: 0.88980144, rec: 0.88069624, f1: 0.8852254

CoNLL Eval:

processed 152728 tokens with 11257 phrases; found: 11227 phrases; correct: 9876.
accuracy:  97.64%; 9876 11257 11227 precision:  87.97%; recall:  87.73%; FB1:  87.85
CARDINAL:  789  935  937 precision:  84.20%; recall:  84.39%; FB1:  84.29  937
DATE:  1399  1602  1640 precision:  85.30%; recall:  87.33%; FB1:  86.30  1640
EVENT:   30   63   43 precision:  69.77%; recall:  47.62%; FB1:  56.60  43
FAC:   72  135  115 precision:  62.61%; recall:  53.33%; FB1:  57.60  115
GPE:  2131  2240  2252 precision:  94.63%; recall:  95.13%; FB1:  94.88  2252
LANGUAGE:    8   22    9 precision:  88.89%; recall:  36.36%; FB1:  51.61  9
LAW:   20   40   31 precision:  64.52%; recall:  50.00%; FB1:  56.34  31
LOC:  123  179  202 precision:  60.89%; recall:  68.72%; FB1:  64.57  202
MONEY:  286  314  321 precision:  89.10%; recall:  91.08%; FB1:  90.08  321
NORP:  803  841  918 precision:  87.47%; recall:  95.48%; FB1:  91.30  918
ORDINAL:  177  195  218 precision:  81.19%; recall:  90.77%; FB1:  85.71  218
ORG:  1502  1795  1687 precision:  89.03%; recall:  83.68%; FB1:  86.27  1687
PERCENT:  306  349  344 precision:  88.95%; recall:  87.68%; FB1:  88.31  344
PERSON:  1887  1988  2020 precision:  93.42%; recall:  94.92%; FB1:  94.16  2020
PRODUCT:   48   76   62 precision:  77.42%; recall:  63.16%; FB1:  69.57  62
QUANTITY:   85  105  111 precision:  76.58%; recall:  80.95%; FB1:  78.70  111
TIME:  128  212  190 precision:  67.37%; recall:  60.38%; FB1:  63.68  190
WORK_OF_ART:   82  166  127 precision:  64.57%; recall:  49.40%; FB1:  55.97  127
```