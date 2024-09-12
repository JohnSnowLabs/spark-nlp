---
layout: model
title: Arabic ElectraForQuestionAnswering model (from aymanm419) Version-2
author: John Snow Labs
name: electra_qa_araElectra_SQUAD_ARCD_768
date: 2024-09-12
tags: [ar, open_source, electra, question_answering, onnx]
task: Question Answering
language: ar
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `araElectra-SQUAD-ARCD-768` is a Arabic model originally trained by `aymanm419`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/electra_qa_araElectra_SQUAD_ARCD_768_ar_5.5.0_3.0_1726122002777.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/electra_qa_araElectra_SQUAD_ARCD_768_ar_5.5.0_3.0_1726122002777.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = MultiDocumentAssembler() \
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = BertForQuestionAnswering.pretrained("electra_qa_araElectra_SQUAD_ARCD_768","ar") \
.setInputCols(["document_question", "document_context"]) \
.setOutputCol("answer")\
.setCaseSensitive(True)

pipeline = Pipeline(stages=[documentAssembler, spanClassifier])

data = spark.createDataFrame([["ما هو اسمي؟", "اسمي كلارا وأنا أعيش في بيركلي."]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new MultiDocumentAssembler() 
.setInputCols(Array("question", "context")) 
.setOutputCols(Array("document_question", "document_context"))

val spanClassifer = BertForQuestionAnswering.pretrained("electra_qa_araElectra_SQUAD_ARCD_768","ar") 
.setInputCols(Array("document", "token")) 
.setOutputCol("answer")
.setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler, spanClassifier))

val data = Seq("ما هو اسمي؟", "اسمي كلارا وأنا أعيش في بيركلي.").toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("ar.answer_question.squad_arcd.electra.768d").predict("""ما هو اسمي؟|||"اسمي كلارا وأنا أعيش في بيركلي.""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|electra_qa_araElectra_SQUAD_ARCD_768|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|ar|
|Size:|504.3 MB|

## References

References

- https://huggingface.co/aymanm419/araElectra-SQUAD-ARCD-768