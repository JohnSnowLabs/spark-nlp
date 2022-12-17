---
layout: model
title: Legal General Provisions Document Classifier (Bert Sentence Embeddings)
author: John Snow Labs
name: legclf_general_provisions_bert
date: 2022-12-16
tags: [en, legal, classification, licensed, bert, general, provisions, tensorflow]
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

The `legclf_general_provisions_bert` model is a Bert Sentence Embeddings Document Classifier used to classify if the document belongs to the class `general-provisions` (check [Lawinsider](https://www.lawinsider.com/tags) for similar document type classification) or not (Binary Classification).

Unlike the Longformer model, this model is lighter in terms of inference time.

## Predicted Entities

`general-provisions`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_general_provisions_bert_en_1.0.0_3.0_1671227646589.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

doc_classifier = legal.ClassifierDLModel.pretrained("legclf_general_provisions_bert", "en", "legal/models")\
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
|[general-provisions]|
|[other]|
|[other]|
|[general-provisions]|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_general_provisions_bert|
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
                    precision    recall  f1-score   support

general-provisions       1.00      0.81      0.90        27
             other       0.89      1.00      0.94        39

          accuracy                           0.92        66
         macro avg       0.94      0.91      0.92        66
      weighted avg       0.93      0.92      0.92        66
```