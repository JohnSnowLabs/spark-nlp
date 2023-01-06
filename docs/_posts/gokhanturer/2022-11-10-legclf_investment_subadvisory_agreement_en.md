---
layout: model
title: Legal Investment Subadvisory Agreement Document Classifier (Longformer)
author: John Snow Labs
name: legclf_investment_subadvisory_agreement
date: 2022-11-10
tags: [en, legal, classification, agreement, investment_subadvisory, licensed]
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

The `legclf_investment_subadvisory_agreement` model is a Legal Longformer Document Classifier to classify if the document belongs to the class `investment-subadvisory-agreement` (check [Lawinsider](https://www.lawinsider.com/tags) for similar document type classification) or not (Binary Classification).

Longformers have a restriction on 4096 tokens, so only the first 4096 tokens will be taken into account. We have realised that for the big majority of the documents in legal corpora, if they are clean and only contain the legal document without any extra information before, 4096 is enough to perform Document Classification.

If not, let us know and we can carry out another approach for you: getting chunks of 4096 tokens and average the embeddings, training with the averaged version, what means all document will be taken into account. But this theoretically should not be required.

## Predicted Entities

`investment-subadvisory-agreement`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_investment_subadvisory_agreement_en_1.0.0_3.0_1668115303552.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    
doc_classifier = nlp.ClassifierDLModel.pretrained("legclf_investment_subadvisory_agreement", "en", "legal/models")\
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
| result|
+-------+
|[investment-subadvisory-agreement]|
|[other]|
|[other]|
|[investment-subadvisory-agreement]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_investment_subadvisory_agreement|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|21.0 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents + Lawinsider categorization

## Benchmarking

```bash

                           label precision recall  f1-score  support
investment-subadvisory-agreement      1.00   0.98      0.99       42
                           other      0.99   1.00      0.99       66
                        accuracy         -      -      0.99      108
                       macro-avg      0.99   0.99      0.99      108
                    weighted-avg      0.99   0.99      0.99      108

```
