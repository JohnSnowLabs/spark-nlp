---
layout: model
title: Turkish deprem_mdeberta_ner DeBertaForTokenClassification from ctoraman
author: John Snow Labs
name: deprem_mdeberta_ner
date: 2024-08-31
tags: [tr, open_source, onnx, token_classification, deberta, ner]
task: Named Entity Recognition
language: tr
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: DeBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DeBertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`deprem_mdeberta_ner` is a Turkish model originally trained by ctoraman.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deprem_mdeberta_ner_tr_5.4.2_3.0_1725114850801.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deprem_mdeberta_ner_tr_5.4.2_3.0_1725114850801.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier  = DeBertaForTokenClassification.pretrained("deprem_mdeberta_ner","tr") \
     .setInputCols(["documents","token"]) \
     .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, tokenClassifier])
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

val tokenClassifier = DeBertaForTokenClassification.pretrained("deprem_mdeberta_ner", "tr")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deprem_mdeberta_ner|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|tr|
|Size:|787.5 MB|

## References

https://huggingface.co/ctoraman/deprem-mdeberta-ner