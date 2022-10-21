---
layout: model
title: Legal Employment agreement Document Binary Classifier
author: John Snow Labs
name: legclf_employment_agreement_document
date: 2022-10-21
tags: [en, legal, classification, document, agreement, contract, licensed]
task: Text Classification
language: en
edition: Spark NLP for Legal 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The `legclf_employment_agreement_document` model is a Legal Longformer Document Classifier to classify if the document belongs to the class employment-agreement (check [Lawinsider](https://www.lawinsider.com/tags) for similar document type classification) or not (Binary Classification).

Longformers have a restriction on 4096 tokens, so only the first 4096 tokens will be taken into account. We have realised that for the big majority of the documents in legal corpora, if they are clean and only contain the legal document without any extra information before, 4096 is enough to perform Document Classification.

If not, let us know and we can carry out another approach for you: getting chunks of 4096 tokens and average the embeddings, training with the averaged version, what means all document will be taken into account. But this theoretically should not be required.

## Predicted Entities

`other`, `employment-agreement`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_employment_agreement_document_en_1.0.0_3.0_1666368638771.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler() \
     .setInputCol("clause_text") \
     .setOutputCol("document")
  
embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")

docClassifier = ClassifierDLModel.pretrained("legclf_employment_agreement_document", "en", "legal/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")
    
nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    embeddings,
    docClassifier])
 
df = spark.createDataFrame([["YOUR TEXT HERE"]]).toDF("clause_text")
model = nlpPipeline.fit(df)
result = model.transform(df)
```

</div>

## Results

```bash
+-------+
| result|
+-------+
|[employment-agreement]|
|[other]|
|[other]|
|[employment-agreement]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_employment_agreement_document|
|Compatibility:|Spark NLP for Legal 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|20.7 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents + Lawinsider categorization

## Benchmarking

```bash
                      precision    recall  f1-score   support

employment-agreement       1.00      1.00      1.00        30
               other       1.00      1.00      1.00        31

            accuracy                           1.00        61
           macro avg       1.00      1.00      1.00        61
        weighted avg       1.00      1.00      1.00        61

```