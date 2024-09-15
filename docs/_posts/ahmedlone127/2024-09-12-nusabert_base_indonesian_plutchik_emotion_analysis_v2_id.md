---
layout: model
title: Indonesian nusabert_base_indonesian_plutchik_emotion_analysis_v2 BertForSequenceClassification from Aardiiiiy
author: John Snow Labs
name: nusabert_base_indonesian_plutchik_emotion_analysis_v2
date: 2024-09-12
tags: [id, open_source, onnx, sequence_classification, bert]
task: Text Classification
language: id
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nusabert_base_indonesian_plutchik_emotion_analysis_v2` is a Indonesian model originally trained by Aardiiiiy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nusabert_base_indonesian_plutchik_emotion_analysis_v2_id_5.5.0_3.0_1726104118926.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nusabert_base_indonesian_plutchik_emotion_analysis_v2_id_5.5.0_3.0_1726104118926.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier  = BertForSequenceClassification.pretrained("nusabert_base_indonesian_plutchik_emotion_analysis_v2","id") \
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

val sequenceClassifier = BertForSequenceClassification.pretrained("nusabert_base_indonesian_plutchik_emotion_analysis_v2", "id")
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
|Model Name:|nusabert_base_indonesian_plutchik_emotion_analysis_v2|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|id|
|Size:|414.6 MB|

## References

https://huggingface.co/Aardiiiiy/NusaBERT-base-Indonesian-Plutchik-emotion-analysis-v2