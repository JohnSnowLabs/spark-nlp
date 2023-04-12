---
layout: model
title: Russian T5ForConditionalGeneration Base Cased model (from 0x7194633)
author: John Snow Labs
name: t5_keyt5_base
date: 2023-01-30
tags: [ru, open_source, t5, tensorflow]
task: Text Generation
language: ru
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

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `keyt5-base` is a Russian model originally trained by `0x7194633`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_keyt5_base_ru_4.3.0_3.0_1675104774932.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_keyt5_base_ru_4.3.0_3.0_1675104774932.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_keyt5_base","ru") \
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
       
val t5 = T5Transformer.pretrained("t5_keyt5_base","ru") 
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
|Model Name:|t5_keyt5_base|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[t5]|
|Language:|ru|
|Size:|927.4 MB|

## References

- https://huggingface.co/0x7194633/keyt5-base
- https://github.com/0x7o/text2keywords
- https://github.com/0x7o/text2keywords
- https://github.com/0x7o/text2keywords
- https://github.com/0x7o/text2keywords
- https://colab.research.google.com/github/0x7o/text2keywords/blob/main/example/keyT5_use.ipynb
- https://colab.research.google.com/github/0x7o/text2keywords/blob/main/example/keyT5_train.ipynb