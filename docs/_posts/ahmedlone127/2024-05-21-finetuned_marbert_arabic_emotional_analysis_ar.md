---
layout: model
title: Arabic finetuned_marbert_arabic_emotional_analysis BertForSequenceClassification from TheKnight115
author: John Snow Labs
name: finetuned_marbert_arabic_emotional_analysis
date: 2024-05-21
tags: [ar, open_source, onnx, sequence_classification, bert]
task: Text Classification
language: ar
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_marbert_arabic_emotional_analysis` is a Arabic model originally trained by TheKnight115.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_marbert_arabic_emotional_analysis_ar_5.2.4_3.0_1716297395902.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_marbert_arabic_emotional_analysis_ar_5.2.4_3.0_1716297395902.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier  = BertForSequenceClassification.pretrained("finetuned_marbert_arabic_emotional_analysis","ar") \
     .setInputCols(["token","document"]) \
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
    .setInputCols("document")
    .setOutputCol("token")

val sequenceClassifier = BertForSequenceClassification.pretrained("finetuned_marbert_arabic_emotional_analysis", "ar")
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
|Model Name:|finetuned_marbert_arabic_emotional_analysis|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|ar|
|Size:|608.8 MB|

## References

https://huggingface.co/TheKnight115/Finetuned_MarBERT_Arabic_Emotional_Analysis