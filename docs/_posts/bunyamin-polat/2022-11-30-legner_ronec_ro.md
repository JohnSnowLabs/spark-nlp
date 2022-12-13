---
layout: model
title: Legal Romanian NER (RONEC dataset)
author: John Snow Labs
name: legner_ronec
date: 2022-11-30
tags: [ro, ner, legal, ronec, licensed]
task: Named Entity Recognition
language: ro
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The `legner_ronec` is a Named Entity Recognition model trained on RONEC (ROmanian Named Entity Corpus). Unlike the original dataset, it has been trained with the following classes:

- PERSON - proper nouns or pronouns if they refer to a person
- LOC - location or geo political entity
- ORG - organization
- LANGUAGE - language
- NAT_REL_POL - national, religious or political organizations
- DATETIME - a time and date in any format, including references to time (e.g. 'yesterday')
- MONEY - a monetary value, numeric or otherwise
- NUMERIC - a simple numeric value, represented as digits or words
- ORDINAL - an ordinal value like 'first', 'third', etc.
- WORK_OF_ART - a work of art like a named TV show, painting, etc.
- EVENT - a named recognizable or periodic major event

## Predicted Entities

`DATETIME`, `EVENT`, `LANGUAGE`, `LOC`, `MONEY`, `NAT_REL_POL`, `NUMERIC`, `ORDINAL`, `ORG`, `PERSON`, `WORK_OF_ART`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_ronec_ro_1.0.0_3.0_1669842840646.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_ronec_ro_1.0.0_3.0_1669842840646.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_base_cased", "ro")\
    .setInputCols("sentence", "token")\
    .setOutputCol("embeddings")\
    .setMaxSentenceLength(512)\
    .setCaseSensitive(True)

ner_model = legal.NerModel.pretrained("legner_ronec", "ro", "legal/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter   
    ])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame([["""Guvernul de stânga italian, condus de premierul Romano Prodi, a devenit după numirea a încă trei secretari de stat, cel mai numeros Executiv din istoria Republicii italiene, având 102 membri."""]]).toDF("text")

result = model.transform(data)
```

</div>

## Results

```bash
+----------------------+-----------+
|ner_chunk             |label      |
+----------------------+-----------+
|Guvernul              |ORG        |
|italian               |NAT_REL_POL|
|premierul Romano Prodi|PERSON     |
|trei                  |NUMERIC    |
|secretari             |PERSON     |
|Republicii italiene   |LOC        |
|102                   |NUMERIC    |
|membri                |PERSON     |
+----------------------+-----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_ronec|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|ro|
|Size:|16.2 MB|

## References

Dataset is available [here](https://github.com/dumitrescustefan/ronec).

## Benchmarking

```bash
label         precision  recall  f1-score  support

DATETIME      0.90       0.90    0.90      1070
EVENT         0.53       0.68    0.59      116
LANGUAGE      0.98       0.95    0.97      44
LOC           0.91       0.90    0.91      1699
MONEY         0.97       0.97    0.97      130
NAT_REL_POL   0.92       0.94    0.93      510
NUMERIC       0.95       0.95    0.95      970
ORDINAL       0.88       0.93    0.90      183
ORG           0.81       0.83    0.82      779
PERSON        0.89       0.91    0.90      2635
WORK_OF_ART   0.73       0.57    0.64      140

micro-avg     0.89       0.90    0.89      8276
macro-avg     0.86       0.87    0.86      8276
weighted-avg  0.89       0.90    0.89      8276
```
