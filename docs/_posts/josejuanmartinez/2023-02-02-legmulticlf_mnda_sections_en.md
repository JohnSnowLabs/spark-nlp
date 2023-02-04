---
layout: model
title: Multilabel Classification of NDA Clauses (small)
author: John Snow Labs
name: legmulticlf_mnda_sections
date: 2023-02-02
tags: [nda, en, licensed, tensorflow]
task: Text Classification
language: en
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

This models should be run on each paragraph of the NDA clauses, and will retrieve a series of 1..N labels for each of them. The possible clause types detected my this model in NDA / MNDA aggrements are:

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


embeddings = (
    nlp.UniversalSentenceEncoder.pretrained()
    .setInputCols("document")
    .setOutputCol("sentence_embeddings")
)

classsifierdl_pred = nlp.MultiClassifierDLModel.pretrained('legmulticlf_mnda_sections', 'en', 'legal/models')\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("class")

clf_pipeline = nlp.Pipeline(stages=[document_assembler,embeddings, classsifierdl_pred])

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
label precision    recall  f1-score   support
         APPLIC_LAW       0.82      0.90      0.86        20
         ASSIGNMENT       0.92      0.88      0.90        26
   DEF_OF_CONF_INFO       0.85      0.85      0.85        27
      DISPUTE_RESOL       0.74      0.50      0.60        28
         EXCEPTIONS       0.93      0.78      0.85        18
   NAMES_OF_PARTIES       0.85      0.81      0.83        21
           NON_COMP       0.95      0.72      0.82        25
          NON_SOLIC       0.68      0.93      0.79        14
           PREAMBLE       0.53      0.56      0.55        16
           REMEDIES       0.80      0.80      0.80        25
          REQ_DISCL       1.00      0.67      0.80         9
RETURN_OF_CONF_INFO       0.73      0.73      0.73        15
        TERMINATION       0.75      0.43      0.55         7
   USE_OF_CONF_INFO       0.86      0.62      0.72        40
          micro-avg       0.82      0.74      0.77       291
          macro-avg       0.82      0.73      0.76       291
       weighted-avg       0.82      0.74      0.77       291
        samples-avg       0.71      0.74      0.71       291
```