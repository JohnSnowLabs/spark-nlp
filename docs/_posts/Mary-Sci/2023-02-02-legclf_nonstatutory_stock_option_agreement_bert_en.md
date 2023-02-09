---
layout: model
title: Legal Nonstatutory Stock Option Agreement Document Classifier (Bert Sentence Embeddings)
author: John Snow Labs
name: legclf_nonstatutory_stock_option_agreement_bert
date: 2023-02-02
tags: [en, legal, classification, nonstatutory, stock, option, agreement, licensed, bert, tensorflow]
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

The `legclf_nonstatutory_stock_option_agreement_bert` model is a Bert Sentence Embeddings Document Classifier used to classify if the document belongs to the class `nonstatutory-stock-option-agreement` (check [Lawinsider](https://www.lawinsider.com/tags) for similar document type classification) or not (Binary Classification).

Unlike the Longformer model, this model is lighter in terms of inference time.

## Predicted Entities

`nonstatutory-stock-option-agreement`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_nonstatutory_stock_option_agreement_bert_en_1.0.0_3.0_1675360953793.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_nonstatutory_stock_option_agreement_bert_en_1.0.0_3.0_1675360953793.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
doc_classifier = legal.ClassifierDLModel.pretrained("legclf_nonstatutory_stock_option_agreement_bert", "en", "legal/models")\
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
|[nonstatutory-stock-option-agreement]|
|[other]|
|[other]|
|[nonstatutory-stock-option-agreement]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_nonstatutory_stock_option_agreement_bert|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.5 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents + Lawinsider categorization

## Benchmarking

```bash
                              label  precision    recall  f1-score   support
nonstatutory-stock-option-agreement       0.98      0.96      0.97        53
                              other       0.98      0.99      0.99       122
                           accuracy          -         -      0.98       175
                          macro-avg       0.98      0.98      0.98       175
                       weighted-avg       0.98      0.98      0.98       175
```
