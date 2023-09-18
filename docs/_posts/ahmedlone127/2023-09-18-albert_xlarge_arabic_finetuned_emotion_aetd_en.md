---
layout: model
title: English albert_xlarge_arabic_finetuned_emotion_aetd AlbertForSequenceClassification from MahaJar
author: John Snow Labs
name: albert_xlarge_arabic_finetuned_emotion_aetd
date: 2023-09-18
tags: [albert, en, open_source, sequence_classification, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.1.2
spark_version: 3.0
supported: true
engine: onnx
annotator: AlbertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albert_xlarge_arabic_finetuned_emotion_aetd` is a English model originally trained by MahaJar.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_xlarge_arabic_finetuned_emotion_aetd_en_5.1.2_3.0_1695066084097.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_xlarge_arabic_finetuned_emotion_aetd_en_5.1.2_3.0_1695066084097.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
sequenceClassifier = AlbertForSequenceClassification.pretrained("albert_xlarge_arabic_finetuned_emotion_aetd","en") \
            .setInputCols(["documents","token"]) \
            .setOutputCol("class")

pipeline = Pipeline().setStages([document_assembler, sequenceClassifier])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val document_assembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("embeddings")
    
val sequenceClassifier = AlbertForSequenceClassification  
    .pretrained("albert_xlarge_arabic_finetuned_emotion_aetd", "en")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("class") 

val pipeline = new Pipeline().setStages(Array(document_assembler, sequenceClassifier))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_xlarge_arabic_finetuned_emotion_aetd|
|Compatibility:|Spark NLP 5.1.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|220.4 MB|

## References

https://huggingface.co/MahaJar/albert-xlarge-arabic-finetuned-emotion_AETD