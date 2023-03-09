---
layout: model
title: Multilabel Classification of NDA Clauses (sentences, small)
author: John Snow Labs
name: legmulticlf_mnda_sections
date: 2023-02-02
tags: [nda, en, licensed, tensorflow]
task: Text Classification
language: en
nav_key: models
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: MultiClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This models should be run on each sentence of the NDA clauses, and will retrieve a series of 1..N labels for each of them. The possible clause types detected my this model in NDA / MNDA aggrements are:

1. Parties to the Agreement - Names of the Parties Clause  
2. Identification of What Information Is Confidential - Definition of Confidential Information Clause
3. Use of Confidential Information: Permitted Use Clause and Obligations of the Recipient
4. Time Frame of the Agreement - Termination Clause  
5. Return of Confidential Information Clause 
6. Remedies for Breaches of Agreement - Remedies Clause 
7. Non-Solicitation Clause
8. Dispute Resolution Clause  
9. Exceptions Clause  
10. Non-competition clause

## Predicted Entities

`APPLIC_LAW`, `ASSIGNMENT`, `DEF_OF_CONF_INFO`, `DISPUTE_RESOL`, `EXCEPTIONS`, `NAMES_OF_PARTIES`, `NON_COMP`, `NON_SOLIC`, `PREAMBLE`, `REMEDIES`, `REQ_DISCL`, `RETURN_OF_CONF_INFO`, `TERMINATION`, `USE_OF_CONF_INFO`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legmulticlf_mnda_sections_en_1.0.0_3.0_1675361534773.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legmulticlf_mnda_sections_en_1.0.0_3.0_1675361534773.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = (
    nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
)

sentence_splitter = (
    nlp.SentenceDetector()
    .setInputCols(["document"])
    .setOutputCol("sentence")
    .setCustomBounds(["\n"])
)

embeddings = (
    nlp.UniversalSentenceEncoder.pretrained()
    .setInputCols("sentence")
    .setOutputCol("sentence_embeddings")
)

classsifierdl_pred = nlp.MultiClassifierDLModel.pretrained('legmulticlf_mnda_sections', 'en', 'legal/models')\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("class")

clf_pipeline = nlp.Pipeline(stages=[document_assembler, sentence_splitter, embeddings, classsifierdl_pred])

df = spark.createDataFrame([["Governing Law.\nThis Agreement shall be govern..."]]).toDF("text")

res = clf_pipeline.fit(df).transform(df)

res.select('text', 'class.result').show()

res.select('class.result')
```

</div>

## Results

```bash
[APPLIC_LAW]	Governing Law.\nThis Agreement shall be govern...
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legmulticlf_mnda_sections|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|12.9 MB|

## References

In-house MNDA

## Benchmarking

```bash
              label    precision    recall  f1-score   support
         APPLIC_LAW       0.93      0.96      0.95        28
         ASSIGNMENT       0.95      0.91      0.93        22
   DEF_OF_CONF_INFO       0.92      0.80      0.86        30
      DISPUTE_RESOL       0.76      0.89      0.82        28
         EXCEPTIONS       0.77      0.91      0.83        11
   NAMES_OF_PARTIES       0.94      0.88      0.91        33
           NON_COMP       1.00      0.91      0.95        23
          NON_SOLIC       0.88      0.94      0.91        16
           PREAMBLE       0.79      0.85      0.81        26
           REMEDIES       0.91      0.91      0.91        32
          REQ_DISCL       0.92      0.92      0.92        13
RETURN_OF_CONF_INFO       1.00      0.96      0.98        24
        TERMINATION       1.00      0.77      0.87        13
   USE_OF_CONF_INFO       0.85      0.88      0.86        32
          micro-avg       0.89      0.89      0.89       331
          macro-avg       0.90      0.89      0.89       331
       weighted-avg       0.90      0.89      0.89       331
        samples-avg       0.87      0.89      0.88       331
```
