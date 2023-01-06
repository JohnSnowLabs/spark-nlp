---
layout: model
title: Legal Plan And Agreement Of Merger Clause Binary Classifier (Bert)
author: John Snow Labs
name: legclf_plan_and_agreement_of_merger_bert
date: 2022-12-09
tags: [en, legal, plan_and_agreement_of_merger, classification, licensed, agreement, tensorflow]
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

The `legclf_plan_and_agreement_of_merger_bert` model is a Bert Sentence Embeddings Document Classifier used to classify if the document belongs to the class `plan-and-agreement-of-merger` (check [Lawinsider](https://www.lawinsider.com/tags) for similar document type classification) or not (Binary Classification).

Unlike the Longformer model, this model is lighter in terms of inference time.

## Predicted Entities

`plan-and-agreement-of-merger`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_plan_and_agreement_of_merger_bert_en_1.0.0_3.0_1670584972507.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    
doc_classifier = legal.ClassifierDLModel.pretrained("legclf_plan_and_agreement_of_merger_bert", "en", "legal/models")\
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

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_plan_and_agreement_of_merger_bert|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.7 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house

## Benchmarking

```bash
                       label  precision    recall  f1-score   support
                       other       1.00      1.00      1.00        59
plan-and-agreement-of-merger       1.00      1.00      1.00        32
                    accuracy          -         -      1.00        91
                   macro-avg       1.00      1.00      1.00        91
                weighted-avg       1.00      1.00      1.00        91
```
