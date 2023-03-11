---
layout: model
title: Legal TERM NER
author: John Snow Labs
name: legner_definitions
date: 2023-03-03
tags: [en, legal, ner, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Thie NER model was trained on a dataset containing legal definitions extracted from state and federal laws in the United States. The definitions cover a range of legal topics, including criminal law, civil law, and commercial law. Each definition includes the term being defined, the source of the definition (e.g., the specific statute or case), and the definition itself.

## Predicted Entities

`TERM`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_definitions_en_1.0.0_3.0_1677844760707.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_definitions_en_1.0.0_3.0_1677844760707.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
document = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence = nlp.SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentence')

token = nlp.Tokenizer()\
    .setInputCols(['sentence'])\
    .setOutputCol('token')

roberta_embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings") \
    .setMaxSentenceLength(512)
  
loaded_ner_model = legal.NerModel.pretrained("legner_definitions", "en", "legal/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

converter = nlp.NerConverter()\
    .setInputCols(["document", "token", "ner"])\
    .setOutputCol("ner_span")

ner_prediction_pipeline = nlp.Pipeline(stages = [
                                            document,
                                            sentence,
                                            token,
                                            roberta_embeddings,
                                            loaded_ner_model,
                                            converter
                                            ])

df = spark.createDataFrame([['''This Amendment No . 2 to Securities Purchase Agreement ( this " Amendment " ) , dated this 5th day of January , 2018 , is made by and among InfoSonics Corporation , a Maryland corporation ( the " Company " ) , and each purchaser identified on the signature pages hereto ( the " Purchasers " ) .''']]).toDF("text")

model = ner_prediction_pipeline.fit(df)

result = model.transform(df)
```

</div>

## Results

```bash

+----------+------+
|chunk     |entity|
+----------+------+
|Amendment |Term  |
|Company   |Term  |
|Purchasers|Term  |
+----------+------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_definitions|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.5 MB|

## References

In-house dataset

## Benchmarking

```bash
label            precision    recall  f1-score   support
      B-Term       0.93      0.94      0.93      1591
      I-Term       0.90      0.93      0.91      1881
   micro-avg       0.91      0.93      0.92      3472
   macro-avg       0.92      0.93      0.92      3472
weighted-avg       0.91      0.93      0.92      3472
```
