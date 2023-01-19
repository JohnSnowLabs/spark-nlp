---
layout: model
title: Italian BertForQuestionAnswering model (from luigisaetta)
author: John Snow Labs
name: bert_qa_squad_xxl_cased_hub1
date: 2022-06-28
tags: [it, open_source, bert, question_answering]
task: Question Answering
language: it
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `squad_it_xxl_cased_hub1` is a Italian model originally trained by `luigisaetta`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_qa_squad_xxl_cased_hub1_it_4.0.0_3.0_1656413780942.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_qa_squad_xxl_cased_hub1_it_4.0.0_3.0_1656413780942.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = MultiDocumentAssembler() \
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = BertForQuestionAnswering.pretrained("bert_qa_squad_xxl_cased_hub1","it") \
.setInputCols(["document_question", "document_context"]) \
.setOutputCol("answer")\
.setCaseSensitive(True)

pipeline = Pipeline(stages=[documentAssembler, spanClassifier])

data = spark.createDataFrame([["Qual è il mio nome?", "Mi chiamo Clara e vivo a Berkeley."]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new MultiDocumentAssembler() 
.setInputCols(Array("question", "context")) 
.setOutputCols(Array("document_question", "document_context"))

val spanClassifer = BertForQuestionAnswering.pretrained("bert_qa_squad_xxl_cased_hub1","it") 
.setInputCols(Array("document", "token")) 
.setOutputCol("answer")
.setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler, spanClassifier))

val data = Seq("Qual è il mio nome?", "Mi chiamo Clara e vivo a Berkeley.").toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("it.answer_question.squad.bert.xxl_cased").predict("""Qual è il mio nome?|||"Mi chiamo Clara e vivo a Berkeley.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_qa_squad_xxl_cased_hub1|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|it|
|Size:|413.3 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/luigisaetta/squad_it_xxl_cased_hub1
- https://github.com/luigisaetta/nlp-qa-italian/blob/main/train_squad_it_final1.ipynb