---
layout: model
title: Legal Stock Option Agreement Document Binary Classifier (Bert Sentence Embeddings)
author: John Snow Labs
name: legclf_stock_option_agreement_bert
date: 2022-12-18
tags: [en, legal, classification, licensed, document, bert, stock, option, agreement, tensorflow]
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

The `legclf_stock_option_agreement_bert` model is a Bert Sentence Embeddings Document Classifier used to classify if the document belongs to the class `stock-option-agreement` (check [Lawinsider](https://www.lawinsider.com/tags) for similar document type classification) or not (Binary Classification).

Unlike the Longformer model, this model is lighter in terms of inference time.

## Predicted Entities

`stock-option-agreement`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_stock_option_agreement_bert_en_1.0.0_3.0_1671393862273.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

doc_classifier = legal.ClassifierDLModel.pretrained("legclf_stock_option_agreement_bert", "en", "legal/models")\
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
|[stock-option-agreement]|
|[other]|
|[other]|
|[stock-option-agreement]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_stock_option_agreement_bert|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.8 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents + Lawinsider categorization

## Benchmarking

```bash

                  label    precision    recall    f1-score    support 
                  other         0.99      0.99        0.99        204 
 stock-option-agreement         0.97      0.97        0.97        107 
               accuracy            -         -        0.98        311 
              macro-avg         0.98      0.98        0.98        311 
           weighted-avg         0.98      0.98        0.98        311

```