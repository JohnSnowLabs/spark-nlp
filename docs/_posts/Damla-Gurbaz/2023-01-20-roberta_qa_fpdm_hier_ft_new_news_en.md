---
layout: model
title: English RobertaForQuestionAnswering Cased model (from AnonymousSub)
author: John Snow Labs
name: roberta_qa_fpdm_hier_ft_new_news
date: 2023-01-20
tags: [en, open_source, roberta, question_answering, tensorflow]
task: Question Answering
language: en
nav_key: models
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: RoBertaForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForQuestionAnswering  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `fpdm_hier_roberta_FT_new_newsqa` is a English model originally trained by `AnonymousSub`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_qa_fpdm_hier_ft_new_news_en_4.3.0_3.0_1674210874319.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_qa_fpdm_hier_ft_new_news_en_4.3.0_3.0_1674210874319.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
Document_Assembler = MultiDocumentAssembler()\
     .setInputCols(["question", "context"])\
     .setOutputCols(["document_question", "document_context"])

Question_Answering = RoBertaForQuestionAnswering.pretrained("roberta_qa_fpdm_hier_ft_new_news","en")\
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

val Question_Answering = RoBertaForQuestionAnswering.pretrained("roberta_qa_fpdm_hier_ft_new_news","en")
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
|Model Name:|roberta_qa_fpdm_hier_ft_new_news|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|461.5 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/AnonymousSub/fpdm_hier_roberta_FT_new_newsqa