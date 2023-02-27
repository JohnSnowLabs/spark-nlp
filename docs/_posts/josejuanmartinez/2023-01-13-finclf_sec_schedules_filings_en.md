---
layout: model
title: Classify Edgar Financial Filings and Schedules
author: John Snow Labs
name: finclf_sec_schedules_filings
date: 2023-01-13
tags: [sec, filings, schedules, en, licensed, tensorflow]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
recommended: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a `multiclass` model, which analyzes the first 512 tokens of your document and retrieves if it is one of the supported classes (see Predicted entities).

The class `schedule` includes `TO-C`, `13D`, `TO-T`, `14F1`, `14D9`, `14N`, `13G`, `TO-I`, `13E3`.
`3` means SEC's `FORM-3`.
`4` means SEC's `FORM-4`.

## Predicted Entities

`schedule`, `other`, `10-K`, `10-Q`, `3`, `4`, `8-K`, `S-8`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_sec_schedules_filings_en_1.0.0_3.0_1673628989895.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_sec_schedules_filings_en_1.0.0_3.0_1673628989895.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
  
embeddings = nlp.UniversalSentenceEncoder.pretrained()\
  .setInputCols("document") \
  .setOutputCol("sentence_embeddings")
    
doc_classifier = finance.ClassifierDLModel.pretrained("finclf_sec_schedules_filings", "en", "finance/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")
    
nlpPipeline = nlp.Pipeline(stages=[
    document_assembler, 
    embeddings,
    doc_classifier
])
 
text = """SECURITIES AND EXCHANGE COMMISSION
WASHINGTON, DC 20549
SCHEDULE 13D
(Rule 13d-101)
INFORMATION TO BE INCLUDED IN STATEMENTS FILED PURSUANT TO RULE 13d-1(a) 
AND AMENDMENTS THERETO FILED PURSUANT TO RULE 13d-2(a)
Under the Securities Exchange Act of 1934
(Amendment No. 2)*
TILE SHOP HOLDINGS, INC.
(Name of Issuer)
...."""

df = spark.createDataFrame([[text]]).toDF("text")

model = nlpPipeline.fit(df)

result = model.transform(df)
```

</div>

## Results

```bash
['schedule']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_sec_schedules_filings|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[category]|
|Language:|en|
|Size:|22.7 MB|

## References

SEC's Edgar

## Benchmarking

```bash
label precision    recall  f1-score   support
        10-K       0.93      0.90      0.92        42
        10-Q       0.95      0.95      0.95        38
           3       0.62      0.61      0.62        33
           4       0.82      0.78      0.80        54
         8-K       0.86      0.91      0.88        33
         S-8       0.93      0.96      0.95        28
       other       1.00      1.00      1.00       238
    schedule       0.94      0.96      0.95        50
    accuracy          -            -          0.93       516
   macro-avg       0.88      0.88      0.88       516
weighted-avg       0.93      0.93      0.93       516
```