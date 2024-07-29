---
layout: model
title: Multilingual T5ForConditionalGeneration Base Cased model (from sonoisa)
author: John Snow Labs
name: t5_base_english_japanese
date: 2024-07-29
tags: [en, ja, multilingual, open_source, t5, xx, onnx]
task: Text Generation
language: xx
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

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `t5-base-english-japanese` is a Multilingual model originally trained by `sonoisa`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_base_english_japanese_xx_5.4.2_3.0_1722246200348.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_base_english_japanese_xx_5.4.2_3.0_1722246200348.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_base_english_japanese","xx") \
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
       
val t5 = T5Transformer.pretrained("t5_base_english_japanese","xx") 
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
|Model Name:|t5_base_english_japanese|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|xx|
|Size:|520.9 MB|

## References

References

- https://huggingface.co/sonoisa/t5-base-english-japanese
- https://en.wikipedia.org
- https://ja.wikipedia.org
- https://oscar-corpus.com
- http://data.statmt.org/cc-100/
- http://data.statmt.org/cc-100/
- https://github.com/sonoisa/t5-japanese
- https://creativecommons.org/licenses/by-sa/4.0/deed.ja
- http://commoncrawl.org/terms-of-use/