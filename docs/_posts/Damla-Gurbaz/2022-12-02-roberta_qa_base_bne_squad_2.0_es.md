---
layout: model
title: Spanish RobertaForQuestionAnswering Base Cased model (from jamarju)
author: John Snow Labs
name: roberta_qa_base_bne_squad_2.0
date: 2022-12-02
tags: [es, open_source, roberta, question_answering, tensorflow]
task: Question Answering
language: es
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForQuestionAnswering  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `roberta-base-bne-squad-2.0-es` is a Spanish model originally trained by `jamarju`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_qa_base_bne_squad_2.0_es_4.2.4_3.0_1669985918426.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
Document_Assembler = MultiDocumentAssembler()\
     .setInputCols(["question", "context"])\
     .setOutputCols(["document_question", "document_context"])

Question_Answering = RoBertaForQuestionAnswering.pretrained("roberta_qa_base_bne_squad_2.0","es")\
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

val Question_Answering = RoBertaForQuestionAnswering.pretrained("roberta_qa_base_bne_squad_2.0","es")
     .setInputCols(Array("document_question", "document_context"))
     .setOutputCol("answer")
     .setCaseSensitive(True)
    
val pipeline = new Pipeline().setStages(Array(Document_Assembler, Question_Answering))

val data = Seq("What's my name?","My name is Clara and I live in Berkeley.").toDS.toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_qa_base_bne_squad_2.0|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|es|
|Size:|456.3 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/jamarju/roberta-base-bne-squad-2.0-es
- https://github.com/PlanTL-SANIDAD/lm-spanish
- https://github.com/ccasimiro88/TranslateAlignRetrieve