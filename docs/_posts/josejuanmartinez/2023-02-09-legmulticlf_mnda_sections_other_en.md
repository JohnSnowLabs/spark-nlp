---
layout: model
title: Multilabel Classification of NDA Clauses (medium)
author: John Snow Labs
name: legmulticlf_mnda_sections_other
date: 2023-02-09
tags: [mnda, nda, en, licensed, tensorflow]
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

This models is a version of `legmulticlf_mnda_sections` (medium) but including more negative examples (OTHER) to reinforce difference between groups and returning `OTHER` also as synonym to `[]`.

It should be run on sentences of the NDA clauses, and will retrieve a series of 1..N labels for each of them. The possible clause types detected my this model in NDA / MNDA aggrements are:

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
11. Other: Nothing of the above (synonym to `[]`)-

## Predicted Entities

`APPLIC_LAW`, `ASSIGNMENT`, `DEF_OF_CONF_INFO`, `DISPUTE_RESOL`, `EXCEPTIONS`, `NAMES_OF_PARTIES`, `NON_COMP`, `NON_SOLIC`, `PREAMBLE`, `REMEDIES`, `REQ_DISCL`, `RETURN_OF_CONF_INFO`, `TERMINATION`, `USE_OF_CONF_INFO`, `OTHER`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legmulticlf_mnda_sections_other_en_1.0.0_3.0_1675938628942.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legmulticlf_mnda_sections_other_en_1.0.0_3.0_1675938628942.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    .setInputCols("document")
    .setOutputCol("sentence_embeddings")
)

classsifierdl_pred = nlp.MultiClassifierDLModel.pretrained('legmulticlf_mnda_sections_other', 'en', 'legal/models')\
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
|Model Name:|legmulticlf_mnda_sections_other|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|13.0 MB|

## References

In-house MNDA

## Benchmarking

```bash
          label          precision    recall  f1-score   support
         APPLIC_LAW       0.90      0.86      0.88        22
         ASSIGNMENT       0.90      0.90      0.90        21
   DEF_OF_CONF_INFO       0.83      0.80      0.82        25
      DISPUTE_RESOL       0.84      0.75      0.79        36
         EXCEPTIONS       0.94      0.85      0.89        20
   NAMES_OF_PARTIES       0.85      0.95      0.90        37
           NON_COMP       0.89      0.94      0.92        18
          NON_SOLIC       0.90      1.00      0.95         9
              OTHER       0.95      0.90      0.92       123
           PREAMBLE       0.88      0.81      0.84        36
           REMEDIES       0.74      0.74      0.74        27
          REQ_DISCL       0.86      0.75      0.80        16
RETURN_OF_CONF_INFO       0.85      0.88      0.87        26
        TERMINATION       0.79      0.79      0.79        19
   USE_OF_CONF_INFO       0.79      0.71      0.75        31
          micro-avg       0.88      0.85      0.86       466
          macro-avg       0.86      0.84      0.85       466
       weighted-avg       0.88      0.85      0.86       466
        samples-avg       0.83      0.85      0.83       466
```
