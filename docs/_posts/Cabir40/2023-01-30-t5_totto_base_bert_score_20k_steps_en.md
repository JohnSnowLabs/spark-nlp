---
layout: model
title: English T5ForConditionalGeneration Base Cased model (from Tejas21)
author: John Snow Labs
name: t5_totto_base_bert_score_20k_steps
date: 2023-01-30
tags: [en, open_source, t5, tensorflow]
task: Text Generation
language: en
nav_key: models
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

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Totto_t5_base_BERT_Score_20k_steps` is a English model originally trained by `Tejas21`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_totto_base_bert_score_20k_steps_en_4.3.0_3.0_1675099761482.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_totto_base_bert_score_20k_steps_en_4.3.0_3.0_1675099761482.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_totto_base_bert_score_20k_steps","en") \
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
       
val t5 = T5Transformer.pretrained("t5_totto_base_bert_score_20k_steps","en") 
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
|Model Name:|t5_totto_base_bert_score_20k_steps|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[t5]|
|Language:|en|
|Size:|925.2 MB|

## References

- https://huggingface.co/Tejas21/Totto_t5_base_BERT_Score_20k_steps
- https://github.com/google-research-datasets/ToTTo
- https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html
- https://github.com/google-research/language/tree/master/language/totto
- https://github.com/Tiiiger/bert_score