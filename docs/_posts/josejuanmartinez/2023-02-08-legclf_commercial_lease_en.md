---
layout: model
title: Legal Lease Agreement Document Classifier (Bert Sentence Embeddings)
author: John Snow Labs
name: legclf_commercial_lease
date: 2023-02-08
tags: [commercial, lease, en, licensed, tensorflow]
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

The `legclf_commercial_lease` model is a Bert Sentence Embeddings Document Classifier used to classify if the document belongs to the class `commercial-lease` (check Lawinsider for similar document type classification) or not (Binary Classification).

Unlike the Longformer model, this model is lighter in terms of inference time.

## Predicted Entities

`commercial-lease`, `other`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/CLASSIFY_LEGAL_CLAUSES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_commercial_lease_en_1.0.0_3.0_1675878333648.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_commercial_lease_en_1.0.0_3.0_1675878333648.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
doc_classifier = legal.ClassifierDLModel.pretrained("legclf_commercial_lease", "en", "legal/models")\
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
|[commercial-lease]|
|[other]|
|[other]|
|[commercial-lease]|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_commercial_lease|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[category]|
|Language:|en|
|Size:|22.7 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents + Lawinsider categorization

## Benchmarking

```bash
              label      precision    recall  f1-score   support
commercial-lease       1.00      0.97      0.98        30
           other       0.95      1.00      0.98        21
        accuracy      -            -         0.98        51
       macro-avg       0.98      0.98      0.98        51
    weighted-avg       0.98      0.98      0.98        51
```