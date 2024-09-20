---
layout: model
title: Thai wangchanberta_base_att_spm_uncased_finetuned_thainewspbs CamemBertForSequenceClassification from SiraH
author: John Snow Labs
name: wangchanberta_base_att_spm_uncased_finetuned_thainewspbs
date: 2024-09-03
tags: [th, open_source, onnx, sequence_classification, camembert]
task: Text Classification
language: th
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: CamemBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wangchanberta_base_att_spm_uncased_finetuned_thainewspbs` is a Thai model originally trained by SiraH.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wangchanberta_base_att_spm_uncased_finetuned_thainewspbs_th_5.5.0_3.0_1725325627440.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wangchanberta_base_att_spm_uncased_finetuned_thainewspbs_th_5.5.0_3.0_1725325627440.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier  = CamemBertForSequenceClassification.pretrained("wangchanberta_base_att_spm_uncased_finetuned_thainewspbs","th") \
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

val sequenceClassifier = CamemBertForSequenceClassification.pretrained("wangchanberta_base_att_spm_uncased_finetuned_thainewspbs", "th")
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
|Model Name:|wangchanberta_base_att_spm_uncased_finetuned_thainewspbs|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|th|
|Size:|394.3 MB|

## References

https://huggingface.co/SiraH/wangchanberta-base-att-spm-uncased-finetuned-ThainewsPBS