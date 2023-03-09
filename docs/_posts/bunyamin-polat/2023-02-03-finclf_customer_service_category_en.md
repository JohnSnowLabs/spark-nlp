---
layout: model
title: Categorize Chat Messages from Customer Service
author: John Snow Labs
name: finclf_customer_service_category
date: 2023-02-03
tags: [en, licensed, finance, customer, classification, tensorflow]
task: Text Classification
language: en
nav_key: models
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: FinanceClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Text Classification model that can help you classify a chat message from customer service.

## Predicted Entities

`ACCOUNT`, `CANCELLATION_FEE`, `CONTACT`, `DELIVERY`, `FEEDBACK`, `INVOICE`, `NEWSLETTER`, `ORDER`, `PAYMENT`, `REFUND`, `SHIPPING_ADDRESS`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_customer_service_category_en_1.0.0_3.0_1675417487415.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_customer_service_category_en_1.0.0_3.0_1675417487415.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

embeddings = nlp.UniversalSentenceEncoder.pretrained() \
    .setInputCols("document") \
    .setOutputCol("sentence_embeddings")

docClassifier = finance.ClassifierDLModel.pretrained("finclf_customer_service_category", "en", "finance/models")\
    .setInputCols("sentence_embeddings") \
    .setOutputCol("class")

pipeline = nlp.Pipeline().setStages(
      [
        document_assembler,
        embeddings,
        docClassifier
      ]
    )

empty_data = spark.createDataFrame([[""]]).toDF("text")
model = pipeline.fit(empty_data)
light_model = nlp.LightPipeline(model)

result = light_model.annotate("""can I place an order from Finland?""")

result["class"]
```

</div>

## Results

```bash
['DELIVERY']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_customer_service_category|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.7 MB|

## References

https://github.com/bitext/customer-support-intent-detection-evaluation-dataset

## Benchmarking

```bash
label             precision  recall  f1-score  support 
ACCOUNT           0.99       0.99    0.99      180     
CANCELLATION_FEE  1.00       1.00    1.00      30      
CONTACT           0.98       1.00    0.99      60      
DELIVERY          1.00       1.00    1.00      60      
FEEDBACK          0.97       0.95    0.96      60      
INVOICE           1.00       1.00    1.00      60      
NEWSLETTER        0.94       1.00    0.97      30      
ORDER             1.00       0.99    1.00      120     
OTHER             1.00       0.97    0.98      63      
PAYMENT           0.95       1.00    0.98      60      
REFUND            0.99       0.98    0.98      90      
SHIPPING_ADDRESS  1.00       0.98    0.99      60      
accuracy          -          -       0.99      973     
macro-avg         0.99       0.99    0.99      873     
weighted-avg      0.99       0.99    0.99      873  
```
