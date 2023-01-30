---
layout: model
title: Italian T5ForConditionalGeneration Base Cased model (from aiknowyou)
author: John Snow Labs
name: t5_mt5_base_it_paraphraser
date: 2023-01-30
tags: [it, open_source, t5, tensorflow]
task: Text Generation
language: it
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

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `mt5-base-it-paraphraser` is a Italian model originally trained by `aiknowyou`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_mt5_base_it_paraphraser_it_4.3.0_3.0_1675105866508.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_mt5_base_it_paraphraser_it_4.3.0_3.0_1675105866508.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_mt5_base_it_paraphraser","it") \
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
       
val t5 = T5Transformer.pretrained("t5_mt5_base_it_paraphraser","it") 
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
|Model Name:|t5_mt5_base_it_paraphraser|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[t5]|
|Language:|it|
|Size:|969.5 MB|

## References

- https://huggingface.co/aiknowyou/mt5-base-it-paraphraser
- https://arxiv.org/abs/2010.11934
- https://colab.research.google.com/drive/1DGeF190gJ3DjRFQiwhFuZalp427iqJNQ
- https://gist.github.com/avidale/44cd35bfcdaf8bedf51d97c468cc8001
- http://creativecommons.org/licenses/by-nc-sa/4.0/
- http://creativecommons.org/licenses/by-nc-sa/4.0/