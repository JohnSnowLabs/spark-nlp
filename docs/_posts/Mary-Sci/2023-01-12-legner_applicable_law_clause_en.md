---
layout: model
title: Applicable Law Clause NER Model
author: John Snow Labs
name: legner_applicable_law_clause
date: 2023-01-12
tags: [en, ner, licensed, applicable_law]
task: Named Entity Recognition
language: en
nav_key: models
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a NER model aimed to be used in `applicable_law` clauses to retrieve entities as `APPLIC_LAW`. Make sure you run this model only on `applicable_law` clauses after you filter them using `legclf_applicable_law_cuad` model.

## Predicted Entities

`APPLIC_LAW`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_applicable_law_clause_en_1.0.0_3.0_1673558480167.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_applicable_law_clause_en_1.0.0_3.0_1673558480167.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base", "en") \
        .setInputCols("sentence", "token") \
        .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained("legner_applicable_law_clause", "en", "legal/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = nlp.Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = ["""ELECTRAMECCANICA VEHICLES CORP., an entity incorporated under the laws of the Province of British Columbia, Canada, with an address of Suite 102 East 1st Avenue, Vancouver, British Columbia, Canada, V5T 1A4 ("EMV")""" ]

result = model.transform(spark.createDataFrame([text]).toDF("text"))
```

</div>

## Results

```bash
+----------------------------------------+----------+----------+
|chunk                                   |ner_label |confidence|
+----------------------------------------+----------+----------+
|laws of the Province of British Columbia|APPLIC_LAW|0.95625716|
+----------------------------------------+----------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_applicable_law_clause|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|1.1 MB|

## References

In-house dataset

## Benchmarking

```bash
       label  precision    recall  f1-score   support
B-APPLIC_LAW       0.90      0.89      0.90        84
I-APPLIC_LAW       0.98      0.93      0.96       425
   micro-avg       0.97      0.93      0.95       509
   macro-avg       0.94      0.91      0.93       509
weighted-avg       0.97      0.93      0.95       509
```
