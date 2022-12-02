---
layout: model
title: Russian DistilBertForQuestionAnswering model (from AndrewChar)
author: John Snow Labs
name: distilbert_qa_model_QA_5_epoch_RU
date: 2022-06-08
tags: [ru, open_source, distilbert, question_answering]
task: Question Answering
language: ru
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: DistilBertForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `model-QA-5-epoch-RU` is a Russian model originally trained by `AndrewChar`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_qa_model_QA_5_epoch_RU_ru_4.0.0_3.0_1654728395737.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = MultiDocumentAssembler() \
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = DistilBertForQuestionAnswering.pretrained("distilbert_qa_model_QA_5_epoch_RU","ru") \
.setInputCols(["document_question", "document_context"]) \
.setOutputCol("answer")\
.setCaseSensitive(True)

pipeline = Pipeline(stages=[documentAssembler, spanClassifier])

data = spark.createDataFrame([["Как меня зовут?", "Меня зовут Клара, и я живу в Беркли."]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new MultiDocumentAssembler() 
.setInputCols(Array("question", "context")) 
.setOutputCols(Array("document_question", "document_context"))

val spanClassifer = DistilBertForQuestionAnswering.pretrained("distilbert_qa_model_QA_5_epoch_RU","ru") 
.setInputCols(Array("document", "token")) 
.setOutputCol("answer")
.setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler, spanClassifier))

val data = Seq("Как меня зовут?", "Меня зовут Клара, и я живу в Беркли.").toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ru.answer_question.distil_bert").predict("""Как меня зовут?|||"Меня зовут Клара, и я живу в Беркли.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_qa_model_QA_5_epoch_RU|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|ru|
|Size:|505.7 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/AndrewChar/model-QA-5-epoch-RU