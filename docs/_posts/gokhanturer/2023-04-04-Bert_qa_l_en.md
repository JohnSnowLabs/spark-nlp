---
layout: model
title: English BertForQuestionAnswering Cased model (from Shobhank-iiitdwd)
author: John Snow Labs
name: Bert_qa_l
date: 2023-04-04
tags: [en, open_source, bert, question_answering, tensorflow]
task: Question Answering
language: en
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

Pretrained BertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `BERT-L-QA` is a English model originally trained by `Shobhank-iiitdwd`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/Bert_qa_l_en_4.4.0_3.0_1680611558838.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/Bert_qa_l_en_4.4.0_3.0_1680611558838.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
Document_Assembler = MultiDocumentAssembler()\
     .setInputCols(["question", "context"])\
     .setOutputCols(["document_question", "document_context"])

Question_Answering = BertForQuestionAnswering.pretrained("Bert_qa_l","en")\
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

val Question_Answering = BertForQuestionAnswering.pretrained("Bert_qa_l","en")
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
|Model Name:|Bert_qa_l|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/Shobhank-iiitdwd/BERT-L-QA
- https://haystack.deepset.ai/tutorials/first-qa-system
- https://github.com/deepset-ai/haystack/
- http://deepset.ai/
- https://haystack.deepset.ai/
- https://deepset.ai/german-bert
- https://deepset.ai/germanquad
- https://github.com/deepset-ai/haystack
- https://docs.haystack.deepset.ai
- https://haystack.deepset.ai/community
- https://twitter.com/deepset_ai
- https://www.linkedin.com/company/deepset-ai/
- https://haystack.deepset.ai/community/join
- https://github.com/deepset-ai/haystack/discussions
- https://deepset.ai
- http://www.deepset.ai/jobs
- https://paperswithcode.com/sota?task=Question+Answering&dataset=squad_v2