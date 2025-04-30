---
layout: model
title: English learn_hf_food_not_food_text_classifier_distilbert_base_uncased_jdevking DistilBertForSequenceClassification from JDevKing
author: John Snow Labs
name: learn_hf_food_not_food_text_classifier_distilbert_base_uncased_jdevking
date: 2025-02-07
tags: [en, open_source, onnx, sequence_classification, distilbert]
task: Text Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`learn_hf_food_not_food_text_classifier_distilbert_base_uncased_jdevking` is a English model originally trained by JDevKing.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/learn_hf_food_not_food_text_classifier_distilbert_base_uncased_jdevking_en_5.5.1_3.0_1738918986467.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/learn_hf_food_not_food_text_classifier_distilbert_base_uncased_jdevking_en_5.5.1_3.0_1738918986467.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')
    
tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier  = DistilBertForSequenceClassification.pretrained("learn_hf_food_not_food_text_classifier_distilbert_base_uncased_jdevking","en") \
     .setInputCols(["documents","token"]) \
     .setOutputCol("class")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, sequenceClassifier])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")
    
val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier = DistilBertForSequenceClassification.pretrained("learn_hf_food_not_food_text_classifier_distilbert_base_uncased_jdevking", "en")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("class") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|learn_hf_food_not_food_text_classifier_distilbert_base_uncased_jdevking|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/JDevKing/learn_hf_food_not_food_text_classifier-distilbert-base-uncased