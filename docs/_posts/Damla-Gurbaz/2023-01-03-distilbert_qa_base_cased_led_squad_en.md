---
layout: model
title: English DistilBertForQuestionAnswering Base Cased model
author: John Snow Labs
name: distilbert_qa_base_cased_led_squad
date: 2023-01-03
tags: [en, open_source, distilbert, question_answering, tensorflow]
task: Question Answering
language: en
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `distilbert-base-cased-distilled-squad` is a English model originally trained by HuggingFace.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_qa_base_cased_led_squad_en_4.3.0_3.0_1672766495924.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_qa_base_cased_led_squad_en_4.3.0_3.0_1672766495924.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
Document_Assembler = MultiDocumentAssembler()\
     .setInputCols(["question", "context"])\
     .setOutputCols(["document_question", "document_context"])

Question_Answering = DistilBertForQuestionAnswering.pretrained("distilbert_qa_base_cased_led_squad","en")\
     .setInputCols(["document_question", "document_context"])\
     .setOutputCol("answer")\
     .setCaseSensitive(True)
    
pipeline = Pipeline(stages=[Document_Assembler, Question_Answering])

data = spark.createDataFrame([["What's my name?","My name is Clara and I live in Berkeley."]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)
```
```scala
val Document_Assembler = new MultiDocumentAssembler()
     .setInputCols(Array("question", "context"))
     .setOutputCols(Array("document_question", "document_context"))

val Question_Answering = DistilBertForQuestionAnswering.pretrained("distilbert_qa_base_cased_led_squad","en")
     .setInputCols(Array("document_question", "document_context"))
     .setOutputCol("answer")
     .setCaseSensitive(true)
    
val pipeline = new Pipeline().setStages(Array(Document_Assembler, Question_Answering))

val data = Seq("What's my name?","My name is Clara and I live in Berkeley.").toDS.toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_qa_base_cased_led_squad|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|244.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/distilbert-base-cased-distilled-squad
- https://arxiv.org/abs/1910.01108
- https://arxiv.org/abs/1910.01108
- https://aclanthology.org/2021.acl-long.330.pdf
- https://dl.acm.org/doi/pdf/10.1145/3442188.3445922
- https://yknzhu.wixsite.com/mbweb
- https://en.wikipedia.org/wiki/English_Wikipedia
- https://mlco2.github.io/impact#compute
- https://arxiv.org/abs/1910.09700
- https://arxiv.org/pdf/1910.01108.pdf
- https://arxiv.org/abs/1910.01108
- https://paperswithcode.com/sota?task=Question+Answering&dataset=squad