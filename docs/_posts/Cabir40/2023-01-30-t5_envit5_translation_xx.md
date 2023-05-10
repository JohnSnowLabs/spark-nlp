---
layout: model
title: Multilingual T5ForConditionalGeneration Cased model (from VietAI)
author: John Snow Labs
name: t5_envit5_translation
date: 2023-01-30
tags: [vi, en, open_source, t5, xx, tensorflow]
task: Text Generation
language: xx
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

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `envit5-translation` is a Multilingual model originally trained by `VietAI`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_envit5_translation_xx_4.3.0_3.0_1675101501844.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_envit5_translation_xx_4.3.0_3.0_1675101501844.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_envit5_translation","xx") \
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
       
val t5 = T5Transformer.pretrained("t5_envit5_translation","xx") 
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
|Model Name:|t5_envit5_translation|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[t5]|
|Language:|xx|
|Size:|599.3 MB|

## References

- https://huggingface.co/VietAI/envit5-translation
- https://paperswithcode.com/sota/machine-translation-on-iwslt2015-english-1?p=mtet-multi-domain-translation-for-english
- https://paperswithcode.com/sota/on-phomt?p=mtet-multi-domain-translation-for-english-and
- https://research.vietai.org/mtet/
- https://github.com/VinAIResearch/PhoMT
- https://user-images.githubusercontent.com/44376091/195998681-5860e443-2071-4048-8a2b-873dcee14a72.png