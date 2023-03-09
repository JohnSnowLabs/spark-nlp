---
layout: model
title: Legal Investment Advisory And Management Agreement Document Classifier (Bert Sentence Embeddings)
author: John Snow Labs
name: legclf_investment_advisory_and_management_agreement_bert
date: 2023-01-29
tags: [en, legal, classification, investment, advisory, management, agreement, licensed, bert, tensorflow]
task: Text Classification
language: en
nav_key: models
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: LegalClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The `legclf_investment_advisory_and_management_agreement_bert` model is a Bert Sentence Embeddings Document Classifier used to classify if the document belongs to the class `investment-advisory-and-management-agreement` or not (Binary Classification).

Unlike the Longformer model, this model is lighter in terms of inference time.

## Predicted Entities

`investment-advisory-and-management-agreement`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_investment_advisory_and_management_agreement_bert_en_1.0.0_3.0_1674990433150.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_investment_advisory_and_management_agreement_bert_en_1.0.0_3.0_1674990433150.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python

document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
  
embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en")\
    .setInputCols("document")\
    .setOutputCol("sentence_embeddings")
    
doc_classifier = legal.ClassifierDLModel.pretrained("legclf_investment_advisory_and_management_agreement_bert", "en", "legal/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")
    
nlpPipeline = nlp.Pipeline(stages=[
    document_assembler, 
    embeddings,
    doc_classifier])
 
df = spark.createDataFrame([["YOUR TEXT HERE"]]).toDF("text")

model = nlpPipeline.fit(df)

result = model.transform(df)

```

</div>

## Results

```bash

+-------+
|result|
+-------+
|[investment-advisory-and-management-agreement]|
|[other]|
|[other]|
|[investment-advisory-and-management-agreement]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_investment_advisory_and_management_agreement_bert|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.2 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents 

## Benchmarking

```bash
                                       label  precision    recall  f1-score   support
investment-advisory-and-management-agreement       1.00      0.97      0.99        34
                                       other       0.98      1.00      0.99        53
                                    accuracy          -         -      0.99        87
                                   macro-avg       0.99      0.99      0.99        87
                                weighted-avg       0.99      0.99      0.99        87
```
