---
layout: model
title: Turkish ElectraForQuestionAnswering model (from kuzgunlar)
author: John Snow Labs
name: electra_qa_turkish
date: 2022-06-22
tags: [tr, open_source, electra, question_answering]
task: Question Answering
language: tr
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `electra-turkish-qa` is a Turkish model originally trained by `kuzgunlar`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/electra_qa_turkish_tr_4.0.0_3.0_1655921428307.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/electra_qa_turkish_tr_4.0.0_3.0_1655921428307.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = MultiDocumentAssembler() \
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = BertForQuestionAnswering.pretrained("electra_qa_turkish","tr") \
.setInputCols(["document_question", "document_context"]) \
.setOutputCol("answer")\
.setCaseSensitive(True)

pipeline = Pipeline(stages=[documentAssembler, spanClassifier])

data = spark.createDataFrame([["Benim adım ne?", "Benim adım Clara ve Berkeley'de yaşıyorum."]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new MultiDocumentAssembler() 
.setInputCols(Array("question", "context")) 
.setOutputCols(Array("document_question", "document_context"))

val spanClassifer = BertForQuestionAnswering.pretrained("electra_qa_turkish","tr") 
.setInputCols(Array("document", "token")) 
.setOutputCol("answer")
.setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler, spanClassifier))

val data = Seq("Benim adım ne?", "Benim adım Clara ve Berkeley'de yaşıyorum.").toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("tr.answer_question.electra").predict("""Benim adım ne?|||"Benim adım Clara ve Berkeley'de yaşıyorum.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|electra_qa_turkish|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|tr|
|Size:|412.7 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/kuzgunlar/electra-turkish-qa