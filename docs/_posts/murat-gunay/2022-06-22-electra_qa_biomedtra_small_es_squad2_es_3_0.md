---
layout: model
title: Spanish ElectraForQuestionAnswering model (from hackathon-pln-es)
author: John Snow Labs
name: electra_qa_biomedtra_small_es_squad2
date: 2022-06-22
tags: [es, open_source, electra, question_answering]
task: Question Answering
language: es
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `biomedtra-small-es-squad2-es` is a Spanish model originally trained by `hackathon-pln-es`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/electra_qa_biomedtra_small_es_squad2_es_4.0.0_3.0_1655920306800.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = MultiDocumentAssembler() \
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = BertForQuestionAnswering.pretrained("electra_qa_biomedtra_small_es_squad2","es") \
.setInputCols(["document_question", "document_context"]) \
.setOutputCol("answer")\
.setCaseSensitive(True)

pipeline = Pipeline(stages=[documentAssembler, spanClassifier])

data = spark.createDataFrame([["¿Cuál es mi nombre?", "Mi nombre es Clara y vivo en Berkeley."]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new MultiDocumentAssembler() 
.setInputCols(Array("question", "context")) 
.setOutputCols(Array("document_question", "document_context"))

val spanClassifer = BertForQuestionAnswering.pretrained("electra_qa_biomedtra_small_es_squad2","es") 
.setInputCols(Array("document", "token")) 
.setOutputCol("answer")
.setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler, spanClassifier))

val data = Seq("¿Cuál es mi nombre?", "Mi nombre es Clara y vivo en Berkeley.").toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.answer_question.squadv2.electra.small").predict("""¿Cuál es mi nombre?|||"Mi nombre es Clara y vivo en Berkeley.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|electra_qa_biomedtra_small_es_squad2|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|es|
|Size:|51.7 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/hackathon-pln-es/biomedtra-small-es-squad2-es