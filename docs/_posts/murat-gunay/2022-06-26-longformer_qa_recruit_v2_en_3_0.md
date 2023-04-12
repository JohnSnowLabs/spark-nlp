---
layout: model
title: English LongformerForQuestionAnswering model (from manishiitg) Version-2
author: John Snow Labs
name: longformer_qa_recruit_v2
date: 2022-06-26
tags: [en, open_source, longformer, question_answering]
task: Question Answering
language: en
nav_key: models
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: LongformerForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `longformer-recruit-qa-v2` is a English model originally trained by `manishiitg`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/longformer_qa_recruit_v2_en_4.0.0_3.0_1656255752690.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/longformer_qa_recruit_v2_en_4.0.0_3.0_1656255752690.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = MultiDocumentAssembler() \
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = LongformerForQuestionAnswering.pretrained("longformer_qa_recruit_v2","en") \
.setInputCols(["document_question", "document_context"]) \
.setOutputCol("answer")\
.setCaseSensitive(True)

pipeline = Pipeline(stages=[documentAssembler, spanClassifier])

data = spark.createDataFrame([["What is my name?", "My name is Clara and I live in Berkeley."]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new MultiDocumentAssembler() 
.setInputCols(Array("question", "context")) 
.setOutputCols(Array("document_question", "document_context"))

val spanClassifer = LongformerForQuestionAnswering.pretrained("longformer_qa_recruit_v2","en") 
.setInputCols(Array("document", "token")) 
.setOutputCol("answer")
.setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler, spanClassifier))

val data = Seq("What is my name?", "My name is Clara and I live in Berkeley.").toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.answer_question.longformer.v2").predict("""What is my name?|||"My name is Clara and I live in Berkeley.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|longformer_qa_recruit_v2|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|556.9 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/manishiitg/longformer-recruit-qa-v2