---
layout: model
title: Legal Subadvisory Agreement Clause Binary Classifier (Bert)
author: John Snow Labs
name: legclf_subadvisory_agreement_bert
date: 2022-12-09
tags: [en, legal, subadvisory_agreement, classification, licensed, agreement, tensorflow]
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

The `legclf_subadvisory_agreement_bert` model is a Bert Sentence Embeddings Document Classifier used to classify if the document belongs to the class `subadvisory-agreement` (check [Lawinsider](https://www.lawinsider.com/tags) for similar document type classification) or not (Binary Classification).

Unlike the Longformer model, this model is lighter in terms of inference time.

## Predicted Entities

`subadvisory-agreement`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_subadvisory_agreement_bert_en_1.0.0_3.0_1670584903166.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    
doc_classifier = legal.ClassifierDLModel.pretrained("legclf_subadvisory_agreement_bert", "en", "legal/models")\
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
|Model Name:|legclf_subadvisory_agreement_bert|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.2 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house

## Benchmarking

```bash
                label  precision    recall  f1-score   support
                other       1.00      1.00      1.00        65
subadvisory-agreement       1.00      1.00      1.00        41
             accuracy          -         -      1.00       106
            macro-avg       1.00      1.00      1.00       106
         weighted-avg       1.00      1.00      1.00       106
```
