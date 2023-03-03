---
layout: model
title: Legal Settlement Agreement And Mutual Release Document Classifier (Bert Sentence Embeddings)
author: John Snow Labs
name: legclf_settlement_agreement_and_mutual_release_bert
date: 2023-01-26
tags: [en, legal, classification, settlement, agreement, mutual, licensed, bert, tensorflow]
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

The `legclf_settlement_agreement_and_mutual_release_bert` model is a Bert Sentence Embeddings Document Classifier used to classify if the document belongs to the class `settlement-agreement-and-mutual-release` or not (Binary Classification).

Unlike the Longformer model, this model is lighter in terms of inference time.

## Predicted Entities

`settlement-agreement-and-mutual-release`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_settlement_agreement_and_mutual_release_bert_en_1.0.0_3.0_1674734834409.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_settlement_agreement_and_mutual_release_bert_en_1.0.0_3.0_1674734834409.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
doc_classifier = legal.ClassifierDLModel.pretrained("legclf_settlement_agreement_and_mutual_release_bert", "en", "legal/models")\
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
|[settlement-agreement-and-mutual-release]|
|[other]|
|[other]|
|[settlement-agreement-and-mutual-release]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_settlement_agreement_and_mutual_release_bert|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.6 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents 

## Benchmarking

```bash
                                  label  precision    recall  f1-score   support
                                  other       0.98      1.00      0.99        61
settlement-agreement-and-mutual-release       1.00      0.97      0.99        39
                               accuracy          -         -      0.99       100
                              macro-avg       0.99      0.99      0.99       100
                           weighted-avg       0.99      0.99      0.99       100
```
