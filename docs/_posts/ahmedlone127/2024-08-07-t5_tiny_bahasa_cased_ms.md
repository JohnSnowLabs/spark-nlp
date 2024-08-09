---
layout: model
title: Malay T5ForConditionalGeneration Tiny Cased model (from mesolitica)
author: John Snow Labs
name: t5_tiny_bahasa_cased
date: 2024-08-07
tags: [ms, open_source, t5, onnx]
task: Text Generation
language: ms
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

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `t5-tiny-bahasa-cased` is a Malay model originally trained by `mesolitica`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_tiny_bahasa_cased_ms_5.4.2_3.0_1723031136197.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_tiny_bahasa_cased_ms_5.4.2_3.0_1723031136197.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_tiny_bahasa_cased","ms") \
    .setInputCols("document") \
    .setOutputCol("answers")
    
pipeline = Pipeline(stages=[documentAssembler, t5])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols("text")
      .setOutputCols("document")
       
val t5 = T5Transformer.pretrained("t5_tiny_bahasa_cased","ms") 
    .setInputCols("document")
    .setOutputCol("answers")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_tiny_bahasa_cased|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|ms|
|Size:|114.0 MB|

## References

References

- https://huggingface.co/mesolitica/t5-tiny-bahasa-cased
- https://github.com/huseinzol05/malaya/tree/master/pretrained-model/t5/prepare
- https://github.com/google-research/text-to-text-transfer-transformer
- https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/t5