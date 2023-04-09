---
layout: model
title: Swahili BertForQuestionAnswering Cased model (from innocent-charles)
author: John Snow Labs
name: bert_qa_swahili_question_answer_latest_cased
date: 2023-04-04
tags: [sw, open_source, bert, question_answering, tensorflow]
task: Question Answering
language: sw
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

Pretrained BertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Swahili-question-answer-latest-cased` is a Swahili model originally trained by `innocent-charles`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_qa_swahili_question_answer_latest_cased_sw_4.4.0_3.0_1680596350855.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_qa_swahili_question_answer_latest_cased_sw_4.4.0_3.0_1680596350855.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
Document_Assembler = MultiDocumentAssembler()\
     .setInputCols(["question", "context"])\
     .setOutputCols(["document_question", "document_context"])

Question_Answering = BertForQuestionAnswering.pretrained("bert_qa_swahili_question_answer_latest_cased","sw")\
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

val Question_Answering = BertForQuestionAnswering.pretrained("bert_qa_swahili_question_answer_latest_cased","sw")
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
|Model Name:|bert_qa_swahili_question_answer_latest_cased|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|sw|
|Size:|665.8 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/innocent-charles/Swahili-question-answer-latest-cased
- https://github.com/Neurotech-HQ/Swahili-QA-dataset
- https://blog.neurotech.africa/building-swahili-question-and-answering-with-haystack/
- https://github.com/deepset-ai/haystack/
- https://haystack.deepset.ai
- https://www.linkedin.com/in/innocent-charles/
- https://github.com/innocent-charles
- https://paperswithcode.com/sota?task=Question+Answering&dataset=kenyacorpus