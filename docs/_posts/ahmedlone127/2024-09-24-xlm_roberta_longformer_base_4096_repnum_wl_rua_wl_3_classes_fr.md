---
layout: model
title: French xlm_roberta_longformer_base_4096_repnum_wl_rua_wl_3_classes XlmRoBertaForSequenceClassification from waboucay
author: John Snow Labs
name: xlm_roberta_longformer_base_4096_repnum_wl_rua_wl_3_classes
date: 2024-09-24
tags: [fr, open_source, onnx, sequence_classification, xlm_roberta]
task: Text Classification
language: fr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: XlmRoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_roberta_longformer_base_4096_repnum_wl_rua_wl_3_classes` is a French model originally trained by waboucay.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_longformer_base_4096_repnum_wl_rua_wl_3_classes_fr_5.5.0_3.0_1727170484761.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_longformer_base_4096_repnum_wl_rua_wl_3_classes_fr_5.5.0_3.0_1727170484761.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier  = XlmRoBertaForSequenceClassification.pretrained("xlm_roberta_longformer_base_4096_repnum_wl_rua_wl_3_classes","fr") \
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

val sequenceClassifier = XlmRoBertaForSequenceClassification.pretrained("xlm_roberta_longformer_base_4096_repnum_wl_rua_wl_3_classes", "fr")
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
|Model Name:|xlm_roberta_longformer_base_4096_repnum_wl_rua_wl_3_classes|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|fr|
|Size:|1.1 GB|

## References

https://huggingface.co/waboucay/xlm-roberta-longformer-base-4096-repnum_wl-rua_wl_3_classes