---
layout: model
title: Dispute Clause Binary Classifier
author: John Snow Labs
name: legclf_dispute_clauses_cuad
date: 2023-01-18
tags: [en, licensed, tensorflow]
task: Text Classification
language: en
nav_key: models
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Binary Classifier (True, False) for the `dispute_clause` clause type. To use this model, make sure you provide enough context as an input.

Senteces have been used as positive examples, so better results will be achieved if SetenceDetector is added to the pipeline.

If you have big legal documents, and you want to look for clauses, we recommend you to split the documents using any of the techniques available in our Spark NLP for Legal Workshop Tokenization & Splitting Tutorial.

Take into consideration the embeddings of this model allows up to 512 tokens. If you have more than that, consider splitting in smaller pieces (you can also check the same tutorial link provided above).

This model can be combined with any of the other 300+ Legal Clauses Classifiers you will find in Models Hub, getting as an output a series of True/False values for each of the legal clause model you have added.

## Predicted Entities

`dispute_clause`, `other`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/CLASSIFY_LEGAL_CLAUSES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_dispute_clauses_cuad_en_1.0.0_3.0_1674056674986.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")

docClassifier = legal.ClassifierDLModel() \
    .pretrained("legclf_dispute_clauses_cuad","en","legal/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("is_dispute_clause")

pipeline = nlp.Pipeline() \
    .setStages(
      [
        documentAssembler,
        embeddings,
        docClassifier
      ]
    )

fit_model = pipeline.fit(spark.createDataFrame([[""]]).toDF('text'))
lm = nlp.LightPipeline(fit_model)

pos_example = "24.2 The parties irrevocably agree that the courts of Ohio shall have non-exclusive jurisdiction to settle any dispute or claim that arises out of or in connection with this agreement or its subject matter or formation ( including non - contractual disputes or claims )."

neg_example = "Brokers’ <strong>Fees and Expenses</strong> Except as expressly set forth in the Transaction Documents to the contrary, each party shall pay the fees and expenses of its advisers, counsel, accountants and other experts, if any, and all other expenses incurred by such party incident to the negotiation, preparation, execution, delivery and performance of this Agreement. The Company shall pay all transfer agent fees, stamp taxes and other taxes and duties levied in connection with the delivery of any Warrant Shares to the Purchasers. Steel Pier Capital Advisors, LLC shall be reimbursed its expenses in having the Transaction Documents prepared on behalf of the Company and for its obligations under the Security Agreement in an amount not to exceed $25,000.00."

texts = [
    pos_example,
    neg_example
]

res = lm.annotate(texts)
```

</div>

## Results

```bash
['dispute_clause']
['other']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_dispute_clauses_cuad|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[label]|
|Language:|en|
|Size:|22.9 MB|

## References

Manual annotations of CUAD dataset

## Benchmarking

```bash
label precision    recall  f1-score   support
dispute_clause       1.00      1.00      1.00        61
         other       1.00      1.00      1.00        96
      accuracy         -         -         1.00       157
     macro-avg       1.00      1.00      1.00       157
  weighted-avg       1.00      1.00      1.00       157
```