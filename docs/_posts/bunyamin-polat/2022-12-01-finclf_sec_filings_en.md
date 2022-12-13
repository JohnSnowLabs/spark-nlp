---
layout: model
title: Financial SEC Filings Classifier
author: John Snow Labs
name: finclf_sec_filings
date: 2022-12-01
tags: [en, finance, classification, sec, licensed]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model allows you to classify documents among a list of specific US Security Exchange Commission filings, as : `10-K`, `10-Q`, `8-K`, `S-8`, `3`, `4`, `Other`

**IMPORTANT** : This model works with the first 512 tokens of a document, you don't need to run it in the whole document.

## Predicted Entities

`10-K`, `10-Q`, `8-K`, `S-8`, `3`, `4`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_sec_filings_en_1.0.0_3.0_1669921534523.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_sec_filings_en_1.0.0_3.0_1669921534523.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
doc_classifier = finance.ClassifierDLModel.pretrained("finclf_sec_filings", "en", "finance/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")
    
nlpPipeline = nlp.Pipeline(stages=[
    document_assembler, 
    embeddings,
    doc_classifier
])
 
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
|[10-K]|
|[8-K]|
|[10-Q]|
|[S-8]|
|[3]|
|[4]|
|[other]|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_sec_filings|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.8 MB|

## References

Scrapped filings from SEC

## Benchmarking

```bash
class         precision  recall  f1-score  support

10-K          0.97       0.82    0.89      40
10-Q          0.94       0.94    0.94      35
3             0.80       0.95    0.87      41
4             0.94       0.76    0.84      42
8-K           0.81       0.94    0.87      32
S-8           0.91       0.93    0.92      44
other         0.98       0.98    0.98      41

accuracy                         0.90      275
macro-avg     0.91       0.90    0.90      275
weighted-avg  0.91       0.90    0.90      275
```
