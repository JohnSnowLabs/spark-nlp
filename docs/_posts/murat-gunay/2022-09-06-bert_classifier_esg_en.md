---
layout: model
title: English BertForSequenceClassification Cased model (from nbroad)
author: John Snow Labs
name: bert_classifier_esg
date: 2022-09-06
tags: [en, open_source, bert, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `ESG-BERT` is a English model originally trained by `nbroad`.

## Predicted Entities

`Waste_And_Hazardous_Materials_Management`, `Management_Of_Legal_And_Regulatory_Framework`, `Air_Quality`, `GHG_Emissions`, `Business_Model_Resilience`, `Water_And_Wastewater_Management`, `Systemic_Risk_Management`, `Director_Removal`, `Data_Security`, `Employee_Engagement_Inclusion_And_Diversity`, `Access_And_Affordability`, `Competitive_Behavior`, `Ecological_Impacts`, `Employee_Health_And_Safety`, `Supply_Chain_Management`, `Critical_Incident_Risk_Management`, `Business_Ethics`, `Product_Design_And_Lifecycle_Management`, `Energy_Management`, `Labor_Practices`, `Physical_Impacts_Of_Climate_Change`, `Product_Quality_And_Safety`, `Human_Rights_And_Community_Relations`, `Customer_Welfare`, `Customer_Privacy`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_esg_en_4.1.0_3.0_1662500066001.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_esg_en_4.1.0_3.0_1662500066001.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_esg","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols(Array("text")) 
      .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_esg","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_esg|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|410.5 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/nbroad/ESG-BERT
- https://github.com/mukut03/ESG-BERT