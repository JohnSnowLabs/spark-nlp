---
layout: model
title: Chinese BertForQuestionAnswering model (from qalover)
author: John Snow Labs
name: bert_qa_chinese_pert_large_open_domain_mrc
date: 2022-06-28
tags: [zh, open_source, bert, question_answering]
task: Question Answering
language: zh
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `chinese-pert-large-open-domain-mrc` is a Chinese model originally trained by `qalover`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_qa_chinese_pert_large_open_domain_mrc_zh_4.0.0_3.0_1656413708959.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_qa_chinese_pert_large_open_domain_mrc_zh_4.0.0_3.0_1656413708959.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = MultiDocumentAssembler() \
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = BertForQuestionAnswering.pretrained("bert_qa_chinese_pert_large_open_domain_mrc","zh") \
.setInputCols(["document_question", "document_context"]) \
.setOutputCol("answer")\
.setCaseSensitive(True)

pipeline = Pipeline(stages=[documentAssembler, spanClassifier])

data = spark.createDataFrame([["PUT YOUR QUESTION HERE", "PUT YOUR CONTEXT HERE"]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new MultiDocumentAssembler() 
.setInputCols(Array("question", "context")) 
.setOutputCols(Array("document_question", "document_context"))

val spanClassifer = BertForQuestionAnswering.pretrained("bert_qa_chinese_pert_large_open_domain_mrc","zh") 
.setInputCols(Array("document", "token")) 
.setOutputCol("answer")
.setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler, spanClassifier))

val data = Seq("PUT YOUR QUESTION HERE", "PUT YOUR CONTEXT HERE").toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("zh.answer_question.bert.large").predict("""PUT YOUR QUESTION HERE|||"PUT YOUR CONTEXT HERE""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_qa_chinese_pert_large_open_domain_mrc|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|zh|
|Size:|1.2 GB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/qalover/chinese-pert-large-open-domain-mrc
- https://github.com/dbiir/UER-py/