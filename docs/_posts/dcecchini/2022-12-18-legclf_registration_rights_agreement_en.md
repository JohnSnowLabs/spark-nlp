---
layout: model
title: Legal Registration Rights Agreement Document Binary Classifier (Longformer)
author: John Snow Labs
name: legclf_registration_rights_agreement
date: 2022-12-18
tags: [en, legal, classification, licensed, document, longformer, registration, rights, agreement, tensorflow]
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

The `legclf_registration_rights_agreement` model is a Longformer Document Classifier used to classify if the document belongs to the class `registration-rights-agreement` (check [Lawinsider](https://www.lawinsider.com/tags) for similar document type classification) or not (Binary Classification).

Longformers have a restriction on 4096 tokens, so only the first 4096 tokens will be taken into account. We have realised that for the big majority of the documents in legal corpora, if they are clean and only contain the legal document without any extra information before, 4096 is enough to perform Document Classification.

If your document needs to process more than 4096 tokens, you can try the following: getting chunks of 4096 tokens and average the embeddings, training with the averaged version, what means all document will be taken into account.

## Predicted Entities

`registration-rights-agreement`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_registration_rights_agreement_en_1.0.0_3.0_1671393670292.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_registration_rights_agreement_en_1.0.0_3.0_1671393670292.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

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

doc_classifier = legal.ClassifierDLModel.pretrained("legclf_registration_rights_agreement", "en", "legal/models")\
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
|[registration-rights-agreement]|
|[other]|
|[other]|
|[registration-rights-agreement]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_registration_rights_agreement|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|21.4 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents + Lawinsider categorization

## Benchmarking

```bash

                         label    precision    recall    f1-score    support 
                         other         0.98      0.99        0.98        205 
 registration-rights-agreement         0.98      0.95        0.97        108 
                      accuracy            -         -        0.98        313 
                     macro-avg         0.98      0.97        0.98        313 
                  weighted-avg         0.98      0.98        0.98        313 

```