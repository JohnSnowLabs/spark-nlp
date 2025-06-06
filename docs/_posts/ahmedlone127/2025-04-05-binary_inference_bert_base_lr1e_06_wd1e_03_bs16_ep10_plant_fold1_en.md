---
layout: model
title: English binary_inference_bert_base_lr1e_06_wd1e_03_bs16_ep10_plant_fold1 BertForSequenceClassification from ys7yoo
author: John Snow Labs
name: binary_inference_bert_base_lr1e_06_wd1e_03_bs16_ep10_plant_fold1
date: 2025-04-05
tags: [en, open_source, onnx, sequence_classification, bert]
task: Text Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`binary_inference_bert_base_lr1e_06_wd1e_03_bs16_ep10_plant_fold1` is a English model originally trained by ys7yoo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/binary_inference_bert_base_lr1e_06_wd1e_03_bs16_ep10_plant_fold1_en_5.5.1_3.0_1743821736858.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/binary_inference_bert_base_lr1e_06_wd1e_03_bs16_ep10_plant_fold1_en_5.5.1_3.0_1743821736858.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier  = BertForSequenceClassification.pretrained("binary_inference_bert_base_lr1e_06_wd1e_03_bs16_ep10_plant_fold1","en") \
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

val sequenceClassifier = BertForSequenceClassification.pretrained("binary_inference_bert_base_lr1e_06_wd1e_03_bs16_ep10_plant_fold1", "en")
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
|Model Name:|binary_inference_bert_base_lr1e_06_wd1e_03_bs16_ep10_plant_fold1|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|414.6 MB|

## References

https://huggingface.co/ys7yoo/binary-inference_bert-base_lr1e-06_wd1e-03_bs16_ep10_plant_fold1