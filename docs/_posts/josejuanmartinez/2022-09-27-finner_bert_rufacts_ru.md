---
layout: model
title: Russian Fact Extraction NER
author: John Snow Labs
name: finner_bert_rufacts
date: 2022-09-27
tags: [ru, licensed]
task: Named Entity Recognition
language: ru
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Russian Name Entity Recognition created for the RuREBus (Russian Relation Extraction for Business) shared task. The overall goal of the shared task was to develop business-oriented models capable of relation and/or fact extraction from texts.

A lot of business applications require relation extraction. Although there are a few corpora, that contain texts annotated with relations, all of them are more of an academic nature and differ from typical business applications. There are a few reasons for this.

First, the annotations are quite tight, i.e. almost every sentence contains an annotated relation. In contrast, the business-oriented documents often contain much less annotated examples. There might be one or two annotated examples in the whole document. Second, the existing corpora cover everyday topics (family relationships, birth and death relations, purchase and sale relations, etc). The business applications require other domain-specific relations. 

The goal of the task is to compare the methods for relation extraction in a more close-to-practice way. For these reasons, we suggest using the documents, produced by the Ministry of Economic Development of the Russian Federation.

The corpus contains regional reports and strategic plans. A part of the corpus is annotated with named entities (8 classes) and semantic relations (11 classes). In total there are approximately 300 annotated documents. The annotation schema and the guidelines for annotators can be found in [here](https://github.com/dialogue-evaluation/RuREBus/blob/master/markup_instruction.pdf) (in Russian).

The dataset consists of:

1. A train set with manually annotated named entities and relations. First and second parts of train set are avaliable [here](https://github.com/dialogue-evaluation/RuREBus/tree/master/train_data)

2.  A large corpus (approx. 280 million tokens) of raw free-form documents, produced by the Ministry of Economic Development. These documents come from the same domain as the train and the test set. This data is avaliable [here](https://yadi.sk/d/9uKbo3p0ghdNpQ).

3. A test set without any annotations

The predicted entities are:

MET - Metric (productivity, growth...)
ECO - Economical Entity / Concept (inner market, energy source...)
BIN - 1-time action, binary (happened or not - construction, development, presence, absence...)
CMP - Quantitative Comparision entity (increase, decrease...)
QUA - Qualitative Comparison entity (stable, limited...)
ACT - Activity (Restauration of buildings, Festivities in Cities...)
INT - Institutions (Centers, Departments, etc)
SOC - Social - Social object (Children, Elder people, Workers of X sector, ...)

## Predicted Entities

`MET`, `ECO`, `BIN`, `CMP`, `QUA`, `ACT`, `INT`, `SOC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FIN_NER_RUSSIAN_GOV/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_bert_rufacts_ru_1.0.0_3.0_1664277912886.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencerDL = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl", "ru") \
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

ner_model = finance.BertForTokenClassification.pretrained("finner_bert_rufacts", "en", "finance/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler,
    sentencerDL,
    tokenizer,
    ner_model,
    ner_converter   
    ])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

text_list = ["""В рамках обеспечения использования в деятельности ОМСУ муниципального образования Московской области региональных и муниципальных информационных систем предусматривается решение задач"""]

import pandas as pd

df = spark.createDataFrame(pd.DataFrame({"text" : text_list}))

result = model.transform(df)

result.select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata')).alias("cols")) \
               .select(F.expr("cols['0']").alias("ner_chunk"),
                       F.expr("cols['1']['entity']").alias("label")).show(truncate = False)
```

</div>

## Results

```bash
+--------------------------------------------------+-----+
|ner_chunk                                         |label|
+--------------------------------------------------+-----+
|обеспечения                                       |BIN  |
|ОМСУ муниципального образования                   |INST |
|региональных и муниципальных информационных систем|ECO  |
+--------------------------------------------------+-----+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_bert_rufacts|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|ru|
|Size:|358.7 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

https://github.com/dialogue-evaluation/RuREBus

## Benchmarking

```bash
       label    precision    recall  f1-score   support
       B-MET       0.7440    0.7440    0.7440       250
       I-MET       0.8301    0.7704    0.7991       945
       B-BIN       0.7248    0.7850    0.7537       614
       B-ACT       0.6052    0.5551    0.5791       254
       I-ACT       0.7215    0.6244    0.6695       892
       B-ECO       0.6892    0.6813    0.6852       524
       I-ECO       0.6750    0.6899    0.6824       861
       B-CMP       0.8405    0.8354    0.8379       164
       I-CMP       0.2000    0.0714    0.1053        14
      B-INST       0.7152    0.7019    0.7085       161
      I-INST       0.7560    0.7114    0.7330       440
       B-SOC       0.5547    0.6698    0.6068       212
       I-SOC       0.6178    0.7087    0.6601       381
       B-QUA       0.6167    0.7303    0.6687       152
       I-QUA       0.7333    0.4400    0.5500        25
   micro-avg       0.7107    0.7017    0.7062      5927
   macro-avg       0.6610    0.6337    0.6413      5927
weighted-avg       0.7136    0.7017    0.7059      5927
```