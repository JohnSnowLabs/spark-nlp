---
layout: model
title: Legal Agreement Of Purchase And Sale Document Classifier (Bert Sentence Embeddings)
author: John Snow Labs
name: legclf_agreement_of_purchase_and_sale_bert
date: 2023-01-26
tags: [en, legal, classification, agreement, purchase, sale, licensed, bert, tensorflow]
task: Text Classification
language: en
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

The `legclf_agreement_of_purchase_and_sale_bert` model is a Bert Sentence Embeddings Document Classifier used to classify if the document belongs to the class `agreement-of-purchase-and-sale` or not (Binary Classification).

Unlike the Longformer model, this model is lighter in terms of inference time.

## Predicted Entities

`agreement-of-purchase-and-sale`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_agreement_of_purchase_and_sale_bert_en_1.0.0_3.0_1674731506904.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_agreement_of_purchase_and_sale_bert_en_1.0.0_3.0_1674731506904.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
doc_classifier = legal.ClassifierDLModel.pretrained("legclf_agreement_of_purchase_and_sale_bert", "en", "legal/models")\
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
|[agreement-of-purchase-and-sale]|
|[other]|
|[other]|
|[agreement-of-purchase-and-sale]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_agreement_of_purchase_and_sale_bert|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.5 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents 

## Benchmarking

```bash


                         label  precision    recall  f1-score   support
agreement-of-purchase-and-sale       0.96      0.98      0.97        51
                         other       0.99      0.98      0.99       116
                      accuracy          -         -      0.98       167
                     macro-avg       0.98      0.98      0.98       167
                  weighted-avg       0.98      0.98      0.98       167
                  
```
