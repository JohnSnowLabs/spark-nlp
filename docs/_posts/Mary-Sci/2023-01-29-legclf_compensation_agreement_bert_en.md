---
layout: model
title: Legal Compensation Agreement Document Classifier (Bert Sentence Embeddings)
author: John Snow Labs
name: legclf_compensation_agreement_bert
date: 2023-01-29
tags: [en, legal, classification, compensation, agreement, licensed, bert, tensorflow]
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

The `legclf_compensation_agreement_bert` model is a Bert Sentence Embeddings Document Classifier used to classify if the document belongs to the class `compensation-agreement` or not (Binary Classification).

Unlike the Longformer model, this model is lighter in terms of inference time.

## Predicted Entities

`compensation-agreement`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_compensation_agreement_bert_en_1.0.0_3.0_1674990338214.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_compensation_agreement_bert_en_1.0.0_3.0_1674990338214.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
doc_classifier = legal.ClassifierDLModel.pretrained("legclf_compensation_agreement_bert", "en", "legal/models")\
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
|[compensation-agreement]|
|[other]|
|[other]|
|[compensation-agreement]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_compensation_agreement_bert|
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
compensation-agreement       1.00      0.95      0.97        40
                 other       0.96      1.00      0.98        55
              accuracy          -         -      0.98        95
             macro-avg       0.98      0.97      0.98        95
          weighted-avg       0.98      0.98      0.98        95       
```
