---
layout: model
title: Spanish BertForQuestionAnswering Cased model (from Josue)
author: John Snow Labs
name: Bert_qa_beto_espanhol_squad2
date: 2023-04-04
tags: [es, open_source, bert, question_answering, tensorflow]
task: Question Answering
language: es
edition: Spark NLP 4.4.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `BETO-espanhol-Squad2` is a Spanish model originally trained by `Josue`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/Bert_qa_beto_espanhol_squad2_es_4.4.0_3.0_1680611707079.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/Bert_qa_beto_espanhol_squad2_es_4.4.0_3.0_1680611707079.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
Document_Assembler = MultiDocumentAssembler()\
     .setInputCols(["question", "context"])\
     .setOutputCols(["document_question", "document_context"])

Question_Answering = BertForQuestionAnswering.pretrained("Bert_qa_beto_espanhol_squad2","es")\
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

val Question_Answering = BertForQuestionAnswering.pretrained("Bert_qa_beto_espanhol_squad2","es")
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
|Model Name:|Bert_qa_beto_espanhol_squad2|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|es|
|Size:|410.1 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/Josue/BETO-espanhol-Squad2
- https://github.com/dccuchile/beto
- https://github.com/ccasimiro88/TranslateAlignRetrieve
- https://github.com/dccuchile/beto/blob/master/README.md
- https://github.com/google-research/bert
- https://github.com/josecannete/spanish-corpora
- https://github.com/google-research/bert/blob/master/multilingual.md
- https://github.com/ccasimiro88/TranslateAlignRetrieve
- https://media.giphy.com/media/mCIaBpfN0LQcuzkA2F/giphy.gif
- https://media.giphy.com/media/WT453aptcbCP7hxWTZ/giphy.gif
- https://twitter.com/Josuehu_