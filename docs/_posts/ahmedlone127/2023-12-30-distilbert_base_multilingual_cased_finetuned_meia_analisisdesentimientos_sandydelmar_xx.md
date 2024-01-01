---
layout: model
title: Multilingual distilbert_base_multilingual_cased_finetuned_meia_analisisdesentimientos_sandydelmar DistilBertForSequenceClassification from SandyDelMar
author: John Snow Labs
name: distilbert_base_multilingual_cased_finetuned_meia_analisisdesentimientos_sandydelmar
date: 2023-12-30
tags: [bert, xx, open_source, sequence_classification, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.2.2
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_base_multilingual_cased_finetuned_meia_analisisdesentimientos_sandydelmar` is a Multilingual model originally trained by SandyDelMar.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_multilingual_cased_finetuned_meia_analisisdesentimientos_sandydelmar_xx_5.2.2_3.0_1703896431630.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_base_multilingual_cased_finetuned_meia_analisisdesentimientos_sandydelmar_xx_5.2.2_3.0_1703896431630.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
sequenceClassifier = DistilBertForSequenceClassification.pretrained("distilbert_base_multilingual_cased_finetuned_meia_analisisdesentimientos_sandydelmar","xx")\
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
    
val sequenceClassifier = DistilBertForSequenceClassification.pretrained("distilbert_base_multilingual_cased_finetuned_meia_analisisdesentimientos_sandydelmar","xx")
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
|Model Name:|distilbert_base_multilingual_cased_finetuned_meia_analisisdesentimientos_sandydelmar|
|Compatibility:|Spark NLP 5.2.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[class]|
|Language:|xx|
|Size:|507.6 MB|

## References

https://huggingface.co/SandyDelMar/distilbert-base-multilingual-cased-finetuned-MeIA-AnalisisDeSentimientos