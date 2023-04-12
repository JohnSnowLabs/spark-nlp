---
layout: model
title: Chinese T5ForConditionalGeneration Base Cased model (from shibing624)
author: John Snow Labs
name: t5_mengzi_base_chinese_correction
date: 2023-01-30
tags: [zh, open_source, t5, tensorflow]
task: Text Generation
language: zh
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: T5Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `mengzi-t5-base-chinese-correction` is a Chinese model originally trained by `shibing624`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_mengzi_base_chinese_correction_zh_4.3.0_3.0_1675105223361.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_mengzi_base_chinese_correction_zh_4.3.0_3.0_1675105223361.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_mengzi_base_chinese_correction","zh") \
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
       
val t5 = T5Transformer.pretrained("t5_mengzi_base_chinese_correction","zh") 
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
|Model Name:|t5_mengzi_base_chinese_correction|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[t5]|
|Language:|zh|
|Size:|1.0 GB|

## References

- https://huggingface.co/shibing624/mengzi-t5-base-chinese-correction
- https://github.com/shibing624/pycorrector
- https://github.com/shibing624/pycorrector/tree/master/pycorrector/t5
- https://pan.baidu.com/s/1BV5tr9eONZCI0wERFvr0gQ
- http://nlp.ee.ncu.edu.tw/resource/csc.html
- https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml