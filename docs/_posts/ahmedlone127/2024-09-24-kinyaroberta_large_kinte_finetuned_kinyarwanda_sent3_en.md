---
layout: model
title: English kinyaroberta_large_kinte_finetuned_kinyarwanda_sent3 RoBertaForSequenceClassification from RogerB
author: John Snow Labs
name: kinyaroberta_large_kinte_finetuned_kinyarwanda_sent3
date: 2024-09-24
tags: [en, open_source, onnx, sequence_classification, roberta]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kinyaroberta_large_kinte_finetuned_kinyarwanda_sent3` is a English model originally trained by RogerB.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kinyaroberta_large_kinte_finetuned_kinyarwanda_sent3_en_5.5.0_3.0_1727171180880.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kinyaroberta_large_kinte_finetuned_kinyarwanda_sent3_en_5.5.0_3.0_1727171180880.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier  = RoBertaForSequenceClassification.pretrained("kinyaroberta_large_kinte_finetuned_kinyarwanda_sent3","en") \
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

val sequenceClassifier = RoBertaForSequenceClassification.pretrained("kinyaroberta_large_kinte_finetuned_kinyarwanda_sent3", "en")
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
|Model Name:|kinyaroberta_large_kinte_finetuned_kinyarwanda_sent3|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|409.3 MB|

## References

https://huggingface.co/RogerB/kinyaRoberta-large-kinte-finetuned-kin-sent3