---
layout: model
title: Extract Intent Type from Customer Service Chat Messages
author: John Snow Labs
name: finclf_customer_service_intent_type
date: 2023-02-03
tags: [en, licensed, intent, finance, customer, tensorflow]
task: Text Classification
language: en
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

This is a Text Classification model that can help you classify a chat message from customer service according to intent type.

## Predicted Entities

`cancel_order`, `change_order`, `change_setup_shipping_address`, `check_cancellation_fee`, `check_payment_methods`, `check_refund_policy`, `complaint`, `contact_customer_service`, `contact_human_agent`, `create_edit_switch_account`, `delete_account`, `delivery_options`, `delivery_period`, `get_check_invoice`, `get_refund`, `newsletter_subscription`, `payment_issue`, `place_order`, `recover_password`, `registration_problems`, `review`, `track_order`, `track_refund`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_customer_service_intent_type_en_1.0.0_3.0_1675427852317.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_customer_service_intent_type_en_1.0.0_3.0_1675427852317.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

docClassifier = finance.ClassifierDLModel.pretrained("finclf_customer_service_intent_type", "en", "finance/models")\
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

result = light_model.annotate("""I have a problem with the deletion of my Premium account.""")

result["class"]
```

</div>

## Results

```bash
['delete_account']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_customer_service_intent_type|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.8 MB|

## References

https://github.com/bitext/customer-support-intent-detection-evaluation-dataset

## Benchmarking

```bash
label                          precision  recall  f1-score  support 
cancel_order                   0.88       1.00    0.94      30      
change_order                   1.00       0.90    0.95      30      
change_setup_shipping_address  0.97       0.97    0.97      36      
check_cancellation_fee         0.97       0.97    0.97      30      
check_payment_methods          0.97       0.93    0.95      30      
check_refund_policy            0.97       0.97    0.97      30      
complaint                      0.93       0.93    0.93      30      
contact_customer_service       1.00       1.00    1.00      30      
contact_human_agent            0.97       0.97    0.97      30      
create_edit_switch_account     0.90       0.97    0.93      36      
delete_account                 0.96       0.87    0.91      30      
delivery_options               0.91       1.00    0.95      30      
delivery_period                1.00       0.97    0.98      30      
get_check_invoice              0.92       0.97    0.95      36      
get_refund                     1.00       0.87    0.93      30      
newsletter_subscription        1.00       0.93    0.97      30      
other                          1.00       0.92    0.96      38      
payment_issue                  0.97       1.00    0.98      30      
place_order                    0.97       0.93    0.95      30      
recover_password               0.97       1.00    0.98      30      
registration_problems          1.00       0.97    0.98      30      
review                         0.94       1.00    0.97      30      
track_order                    0.93       0.93    0.93      30      
track_refund                   0.91       1.00    0.95      30      
accuracy                       -          -       0.96      746     
macro-avg                      0.96       0.96    0.96      746     
weighted-avg                   0.96       0.96    0.96      746      
```
