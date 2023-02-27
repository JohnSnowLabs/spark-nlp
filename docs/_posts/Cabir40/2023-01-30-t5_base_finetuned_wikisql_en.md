---
layout: model
title: English T5ForConditionalGeneration Base Cased model (from mrm8488)
author: John Snow Labs
name: t5_base_finetuned_wikisql
date: 2023-01-30
tags: [en, open_source, t5, tensorflow]
task: Text Generation
language: en
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

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `t5-base-finetuned-wikiSQL` is a English model originally trained by `mrm8488`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_base_finetuned_wikisql_en_4.3.0_3.0_1675109286457.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_base_finetuned_wikisql_en_4.3.0_3.0_1675109286457.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_base_finetuned_wikisql","en") \
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
       
val t5 = T5Transformer.pretrained("t5_base_finetuned_wikisql","en") 
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
|Model Name:|t5_base_finetuned_wikisql|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[t5]|
|Language:|en|
|Size:|887.6 MB|

## References

- https://huggingface.co/mrm8488/t5-base-finetuned-wikiSQL
- https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html
- https://github.com/salesforce/WikiSQL
- https://arxiv.org/pdf/1910.10683.pdf
- https://i.imgur.com/jVFMMWR.png
- https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb
- https://github.com/patil-suraj
- https://pbs.twimg.com/media/Ec5vaG5XsAINty_?format=png&name=900x900
- https://twitter.com/mrm8488
- https://www.linkedin.com/in/manuel-romero-cs/