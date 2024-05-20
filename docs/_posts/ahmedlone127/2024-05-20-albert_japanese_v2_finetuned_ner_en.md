---
layout: model
title: English albert_japanese_v2_finetuned_ner AlbertForTokenClassification from ajinkyaT
author: John Snow Labs
name: albert_japanese_v2_finetuned_ner
date: 2024-05-20
tags: [en, open_source, token_classification, albert, ner, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: AlbertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`albert_japanese_v2_finetuned_ner` is a English model originally trained by ajinkyaT.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_japanese_v2_finetuned_ner_en_5.2.4_3.0_1716214142866.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_japanese_v2_finetuned_ner_en_5.2.4_3.0_1716214142866.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier   = AlbertForTokenClassification.pretrained("albert_japanese_v2_finetuned_ner","en") \
     .setInputCols(["token","document"]) \
     .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, tokenClassifier ])
data = spark.createDataFrame([["Saya suka Spark NLP"]]).toDF("text")
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

val tokenClassifier  = AlbertForTokenClassification.pretrained("albert_japanese_v2_finetuned_ner", "en")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier ))
val data = Seq("Où est-ce que je vis?").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_japanese_v2_finetuned_ner|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|43.1 MB|

## References

https://huggingface.co/ajinkyaT/albert-japanese-v2-finetuned-ner