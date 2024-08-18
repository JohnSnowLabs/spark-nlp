---
layout: model
title: English ke_t5_small_finetuned_amazon_english_spanish T5Transformer from alphahg
author: John Snow Labs
name: ke_t5_small_finetuned_amazon_english_spanish
date: 2024-08-18
tags: [en, open_source, onnx, t5, question_answering, summarization, translation, text_generation]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: T5Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ke_t5_small_finetuned_amazon_english_spanish` is a English model originally trained by alphahg.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ke_t5_small_finetuned_amazon_english_spanish_en_5.4.2_3.0_1723985730907.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ke_t5_small_finetuned_amazon_english_spanish_en_5.4.2_3.0_1723985730907.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

t5  = T5Transformer.pretrained("ke_t5_small_finetuned_amazon_english_spanish","en") \
     .setInputCols(["document"]) \
     .setOutputCol("output")

pipeline = Pipeline().setStages([documentAssembler, t5])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")

val t5 = T5Transformer.pretrained("ke_t5_small_finetuned_amazon_english_spanish", "en")
    .setInputCols(Array("documents")) 
    .setOutputCol("output") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ke_t5_small_finetuned_amazon_english_spanish|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|en|
|Size:|493.6 MB|

## References

https://huggingface.co/alphahg/ke-t5-small-finetuned-amazon-en-es