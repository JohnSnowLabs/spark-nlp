---
layout: model
title: Legal Administrative Services Agreement Clause Binary Classifier (Bert)
author: John Snow Labs
name: legclf_administrative_services_agreement_bert
date: 2022-12-09
tags: [en, legal, administrative_services_agreement, classification, licensed, agreement, tensorflow]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The `legclf_administrative_services_agreement_bert` model is a Bert Sentence Embeddings Document Classifier used to classify if the document belongs to the class `administrative-services-agreement` (check [Lawinsider](https://www.lawinsider.com/tags) for similar document type classification) or not (Binary Classification).

Unlike the Longformer model, this model is lighter in terms of inference time.

## Predicted Entities

`administrative-services-agreement`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_administrative_services_agreement_bert_en_1.0.0_3.0_1670584332931.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    
doc_classifier = legal.ClassifierDLModel.pretrained("legclf_administrative_services_agreement_bert", "en", "legal/models")\
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

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_administrative_services_agreement_bert|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.2 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house

## Benchmarking

```bash
                            label  precision    recall  f1-score   support
administrative-services-agreement       1.00      0.88      0.94        26
                            other       0.96      1.00      0.98        65
                         accuracy          -         -      0.97        91
                        macro-avg       0.98      0.94      0.96        91
                     weighted-avg       0.97      0.97      0.97        91
```
