---
layout: model
title: Legal Agreement and Declaration Document Classifier (Longformer)
author: John Snow Labs
name: legclf_agreement_and_declaration
date: 2022-11-10
tags: [licensed, legal, en, classification, agreement, declaration]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The `legclf_agreement_and_declaration` model is a Legal Longformer Document Classifier to classify if the document belongs to the class `agreement-and-declaration` (check Lawinsider for similar document type classification) or not (Binary Classification).

Longformers have a restriction on 4096 tokens, so only the first 4096 tokens will be taken into account. We have realised that for the big majority of the documents in legal corpora, if they are clean and only contain the legal document without any extra information before, 4096 is enough to perform Document Classification.

If not, let us know and we can carry out another approach for you: getting chunks of 4096 tokens and average the embeddings, training with the averaged version, what means all document will be taken into account. But this theoretically should not be required.

## Predicted Entities

`agreement-and-declaration`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_agreement_and_declaration_en_1.0.0_3.0_1668106274229.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler() \
     .setInputCol("text") \
     .setOutputCol("document")
tokenizer = nlp.Tokenizer()\
     .setInputCols(["document"])\
     .setOutputCol("token")
embeddings = nlp.LongformerEmbeddings.pretrained("legal_longformer_base", "en")\
    .setInputCols("document", "token") \
    .setOutputCol("embeddings")
sembeddings = nlp.SentenceEmbeddings()\
    .setInputCols(["document", "embeddings"]) \
    .setOutputCol("sentence_embeddings") \
    .setPoolingStrategy("AVERAGE")
docClassifier = nlp.ClassifierDLModel.pretrained("legclf_agreement_and_declaration", "en", "legal/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")
    
nlpPipeline = nlp.Pipeline(stages=[
    documentAssembler, 
    tokenizer,
    embeddings,
    sembeddings,
    docClassifier])
 
df = spark.createDataFrame([["YOUR TEXT HERE"]]).toDF("text")
model = nlpPipeline.fit(df)
result = model.transform(df)
```

</div>

## Results

```bash
+-------+
| result|
+-------+
|[agreement-and-declaration]|
|[other]|
|[other]|
|[agreement-and-declaration]|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_agreement_and_declaration|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|21.6 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents + Lawinsider categorization

## Benchmarking

```bash

                    label precision recall  f1-score  support
agreement-and-declaration      0.97   1.00      0.98       32
                    other      1.00   0.98      0.99       66
                 accuracy         -      -      0.99       98
                macro-avg      0.98   0.99      0.99       98
             weighted-avg      0.99   0.99      0.99       98

```
