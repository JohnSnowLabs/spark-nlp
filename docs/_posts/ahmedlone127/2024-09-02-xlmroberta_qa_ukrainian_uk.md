---
layout: model
title: Ukrainian XlmRoBertaForQuestionAnswering model (from robinhad)
author: John Snow Labs
name: xlmroberta_qa_ukrainian
date: 2024-09-02
tags: [uk, open_source, xlmroberta, question_answering, onnx]
task: Question Answering
language: uk
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: XlmRoBertaForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `ukrainian-qa` is a Ukrainian model originally trained by `robinhad`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_qa_ukrainian_uk_5.5.0_3.0_1725254362143.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_qa_ukrainian_uk_5.5.0_3.0_1725254362143.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = MultiDocumentAssembler() \
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = XlmRoBertaForQuestionAnswering.pretrained("xlmroberta_qa_ukrainian","uk") \
.setInputCols(["document_question", "document_context"]) \
.setOutputCol("answer")\
.setCaseSensitive(True)

pipeline = Pipeline(stages=[documentAssembler, spanClassifier])

data = spark.createDataFrame([["Як мене звати?", "Мене звуть Клара, і я живу в Берклі."]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new MultiDocumentAssembler() 
.setInputCols(Array("question", "context")) 
.setOutputCols(Array("document_question", "document_context"))

val spanClassifer = XlmRoBertaForQuestionAnswering.pretrained("xlmroberta_qa_ukrainian","uk") 
.setInputCols(Array("document", "token")) 
.setOutputCol("answer")
.setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler, spanClassifier))

val data = Seq("Як мене звати?", "Мене звуть Клара, і я живу в Берклі.").toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("uk.answer_question.xlmr_roberta").predict("""Як мене звати?|||"Мене звуть Клара, і я живу в Берклі.""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_qa_ukrainian|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|uk|
|Size:|401.3 MB|

## References

References

- https://huggingface.co/robinhad/ukrainian-qa
- https://github.com/fido-ai/ua-datasets/tree/main/ua_datasets/src/question_answering
- https://github.com/robinhad/ukrainian-qa