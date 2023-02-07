---
layout: model
title: Italian T5ForConditionalGeneration Small Cased model (from it5)
author: John Snow Labs
name: t5_it5_efficient_small_el32_question_answering
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

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `it5-efficient-small-el32-question-answering` is a Italian model originally trained by `it5`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_it5_efficient_small_el32_question_answering_it_4.3.0_3.0_1675103534190.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_it5_efficient_small_el32_question_answering_it_4.3.0_3.0_1675103534190.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_it5_efficient_small_el32_question_answering","it") \
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
       
val t5 = T5Transformer.pretrained("t5_it5_efficient_small_el32_question_answering","it") 
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
|Model Name:|t5_it5_efficient_small_el32_question_answering|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[t5]|
|Language:|it|
|Size:|593.9 MB|

## References

- https://huggingface.co/it5/it5-efficient-small-el32-question-answering
- https://github.com/stefan-it
- https://arxiv.org/abs/2203.03759
- https://gsarti.com
- https://malvinanissim.github.io
- https://github.com/gsarti/it5
- https://paperswithcode.com/sota?task=Question+Answering&dataset=SQuAD-IT