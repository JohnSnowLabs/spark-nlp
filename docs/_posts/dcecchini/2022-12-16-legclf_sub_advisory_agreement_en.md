---
layout: model
title: Legal Sub Advisory Agreement Document Classifier (Longformer)
author: John Snow Labs
name: legclf_sub_advisory_agreement
date: 2022-12-16
tags: [en, legal, classification, licensed, longformer, sub, advisory, agreement, tensorflow]
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

The `legclf_sub_advisory_agreement` model is a Longformer Document Classifier used to classify if the document belongs to the class `sub-advisory-agreement` (check [Lawinsider](https://www.lawinsider.com/tags) for similar document type classification) or not (Binary Classification).

    Unlike the Longformer model, this model is lighter in terms of inference time.

## Predicted Entities

`sub-advisory-agreement`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_sub_advisory_agreement_en_1.0.0_3.0_1671227705868.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

        document_assembler = nlp.DocumentAssembler()\
            .setInputCol("text")\
            .setOutputCol("document")
            
        tokenizer = nlp.Tokenizer()             .setInputCols(["document"])             .setOutputCol("token")            
        embeddings = nlp.LongformerEmbeddings.pretrained("legal_longformer_base", language)               .setInputCols("document", "token")               .setOutputCol("embeddings")
        
        sentence_embeddings = nlp.SentenceEmbeddings()            .setInputCols(["document", "embeddings"])             .setOutputCol("sentence_embeddings")             .setPoolingStrategy("AVERAGE")

        doc_classifier = legal.ClassifierDLModel.pretrained("legclf_sub_advisory_agreement", "en", "legal/models")\
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
    |[sub-advisory-agreement]|
    |[other]|
    |[other]|
    |[sub-advisory-agreement]|
    
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_sub_advisory_agreement|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.7 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents + Lawinsider categorization

## Benchmarking

```bash

                        precision    recall  f1-score   support

                 other       0.99      1.00      0.99       202
sub-advisory-agreement       1.00      0.97      0.99       104

              accuracy                           0.99       306
             macro avg       0.99      0.99      0.99       306
          weighted avg       0.99      0.99      0.99       306

```