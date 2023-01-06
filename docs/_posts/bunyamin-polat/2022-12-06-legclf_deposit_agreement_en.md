---
layout: model
title: Legal Deposit Agreement Document Classifier (Longformer)
author: John Snow Labs
name: legclf_deposit_agreement
date: 2022-12-06
tags: [en, legal, classification, agreement, deposit, licensed, tensorflow]
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

The `legclf_deposit_agreement` model is a Legal Longformer Document Classifier to classify if the document belongs to the class `deposit-agreement` (check [Lawinsider](https://www.lawinsider.com/tags) for similar document type classification) or not (Binary Classification).

Longformers have a restriction on 4096 tokens, so only the first 4096 tokens will be taken into account. We have realised that for the big majority of the documents in legal corpora, if they are clean and only contain the legal document without any extra information before, 4096 is enough to perform Document Classification.

If not, let us know and we can carry out another approach for you: getting chunks of 4096 tokens and average the embeddings, training with the averaged version, what means all document will be taken into account. But this theoretically should not be required.

## Predicted Entities

`deposit-agreement`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_deposit_agreement_en_1.0.0_3.0_1670357636929.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python

document_assembler = nlp.DocumentAssembler()\
     .setInputCol("text")\
     .setOutputCol("document")\
     
tokenizer = nlp.Tokenizer()\
     .setInputCols(["document"])\
     .setOutputCol("token")
     
embeddings = nlp.LongformerEmbeddings.pretrained("legal_longformer_base", "en")\
    .setInputCols("document", "token")\
    .setOutputCol("embeddings")
    
sentence_embeddings = nlp.SentenceEmbeddings()\
    .setInputCols(["document", "embeddings"])\
    .setOutputCol("sentence_embeddings")\
    .setPoolingStrategy("AVERAGE")
    
doc_classifier = legal.ClassifierDLModel.pretrained("legclf_deposit_agreement", "en", "legal/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")
    
nlpPipeline = nlp.Pipeline(stages=[
    document_assembler, 
    tokenizer,
    embeddings,
    sentence_embeddings,
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
|[deposit-agreement]|
|[other]|
|[other]|
|[deposit-agreement]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_deposit_agreement|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|21.2 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents + Lawinsider categorization

## Benchmarking

```bash

            label   precision    recall  f1-score   support
deposit-agreement       1.00      0.93      0.97        61
            other       0.97      1.00      0.98       111
         accuracy         -         -       0.98       172
        macro-avg       0.98      0.97      0.97       172
     weighted-avg       0.98      0.98      0.98       172

```
