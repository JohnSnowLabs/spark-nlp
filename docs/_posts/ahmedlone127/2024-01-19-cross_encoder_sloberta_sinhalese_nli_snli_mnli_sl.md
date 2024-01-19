---
layout: model
title: Slovenian cross_encoder_sloberta_sinhalese_nli_snli_mnli CamemBertForSequenceClassification from jacinthes
author: John Snow Labs
name: cross_encoder_sloberta_sinhalese_nli_snli_mnli
date: 2024-01-19
tags: [camembert, sl, open_source, sequence_classification, onnx]
task: Text Classification
language: sl
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: CamemBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cross_encoder_sloberta_sinhalese_nli_snli_mnli` is a Slovenian model originally trained by jacinthes.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cross_encoder_sloberta_sinhalese_nli_snli_mnli_sl_5.2.4_3.0_1705699360309.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cross_encoder_sloberta_sinhalese_nli_snli_mnli_sl_5.2.4_3.0_1705699360309.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer()\
    .setInputCols("document")\
    .setOutputCol("token")  
    
sequenceClassifier = CamemBertForSequenceClassification.pretrained("cross_encoder_sloberta_sinhalese_nli_snli_mnli","sl")\
            .setInputCols(["document","token"])\
            .setOutputCol("class")

pipeline = Pipeline().setStages([document_assembler, tokenizer, sequenceClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)

```
```scala

val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document") 
    .setOutputCol("token")  
    
val sequenceClassifier = CamemBertForSequenceClassification.pretrained("cross_encoder_sloberta_sinhalese_nli_snli_mnli","sl")
            .setInputCols(Array("document","token"))
            .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cross_encoder_sloberta_sinhalese_nli_snli_mnli|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[class]|
|Language:|sl|
|Size:|411.9 MB|

## References

https://huggingface.co/jacinthes/cross-encoder-sloberta-si-nli-snli-mnli