---
layout: model
title: Korean T5ForConditionalGeneration Small Cased model (from KETI-AIR)
author: John Snow Labs
name: t5_ke_small
date: 2024-07-14
tags: [ko, open_source, t5, onnx]
task: Text Generation
language: ko
edition: Spark NLP 5.4.1
spark_version: 3.0
supported: true
engine: onnx
annotator: T5Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `ke-t5-small-ko` is a Korean model originally trained by `KETI-AIR`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_ke_small_ko_5.4.1_3.0_1720955364486.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_ke_small_ko_5.4.1_3.0_1720955364486.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_ke_small","ko") \
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
       
val t5 = T5Transformer.pretrained("t5_ke_small","ko") 
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
|Model Name:|t5_ke_small|
|Compatibility:|Spark NLP 5.4.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|ko|
|Size:|273.7 MB|

## References

References

- https://huggingface.co/KETI-AIR/ke-t5-small-ko
- https://github.com/AIRC-KETI/ke-t5
- https://aclanthology.org/2021.findings-emnlp.33/
- https://koreascience.kr/article/CFKO202130060717834.pdf