---
layout: model
title: 10K Item Section Classifier
author: John Snow Labs
name: finclf_10k_items
date: 2023-03-10
tags: [en, licensed, classifier, 10k_items, finance, tensorflow]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: MedicalBertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Multiclass classification model which identifies the item (section) number in a 10K filing.

## Predicted Entities

`section_1`, `section_2`, `section_3`, `section_7`, `section_8`, `section_10`, `section_12`, `section_13`, `section_14`, `section_15`, `section_1A`, `section_1B`, `section_7A`, `section_9A`, `section_9B`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_10k_items_en_1.0.0_3.0_1678450523713.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_10k_items_en_1.0.0_3.0_1678450523713.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = nlp.Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

seq_classifier = finance.BertForSequenceClassification.pretrained("finclf_10k_items", "en", "finance/models") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = nlp.Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["These issues could negatively affect the timely collection of our U.S. government invoices."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```

</div>

## Results

```bash
+------------+
|      result|
+------------+
|[section_10]|
+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_10k_items|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|412.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Train dataset available [here](https://huggingface.co/datasets/JanosAudran/financial-reports-sec)

## Benchmarking

```bash
label         precision  recall  f1-score  support 
section_1     0.59       0.66    0.62      112     
section_10    0.73       0.72    0.72      137     
section_12    0.95       1.00    0.97      124     
section_13    0.93       0.94    0.94      212     
section_14    0.99       0.97    0.98      172     
section_15    0.91       0.84    0.87      139     
section_1A    0.85       0.86    0.85      92      
section_1B    0.70       0.64    0.67      233     
section_2     0.85       0.78    0.81      172     
section_3     0.60       0.69    0.64      224     
section_7     0.92       0.93    0.92      164     
section_7A    0.89       0.90    0.89      99      
section_8     0.80       0.97    0.88      72      
section_9A    0.91       0.93    0.92      75      
section_9B    0.77       0.63    0.69      147     
accuracy      -          -       0.81      2174    
macro-avg     0.83       0.83    0.83      2174    
weighted-avg  0.82       0.81    0.81      2174  
```
