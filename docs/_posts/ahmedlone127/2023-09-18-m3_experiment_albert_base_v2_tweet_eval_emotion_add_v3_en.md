---
layout: model
title: English m3_experiment_albert_base_v2_tweet_eval_emotion_add_v3 AlbertForSequenceClassification from m3
author: John Snow Labs
name: m3_experiment_albert_base_v2_tweet_eval_emotion_add_v3
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

Pretrained AlbertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`m3_experiment_albert_base_v2_tweet_eval_emotion_add_v3` is a English model originally trained by m3.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/m3_experiment_albert_base_v2_tweet_eval_emotion_add_v3_en_5.1.2_3.0_1695063651684.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/m3_experiment_albert_base_v2_tweet_eval_emotion_add_v3_en_5.1.2_3.0_1695063651684.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
sequenceClassifier = AlbertForSequenceClassification.pretrained("m3_experiment_albert_base_v2_tweet_eval_emotion_add_v3","en") \
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
    .pretrained("m3_experiment_albert_base_v2_tweet_eval_emotion_add_v3", "en")
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
|Model Name:|m3_experiment_albert_base_v2_tweet_eval_emotion_add_v3|
|Compatibility:|Spark NLP 5.1.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|44.2 MB|

## References

https://huggingface.co/m3/m3-experiment-albert-base-v2-tweet-eval-emotion-add-v3