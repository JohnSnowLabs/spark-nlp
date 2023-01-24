---
layout: model
title: Arabic ElectraForQuestionAnswering model (from Damith)
author: John Snow Labs
name: electra_qa_AraELECTRA_discriminator_SOQAL
date: 2022-06-22
tags: [ar, open_source, electra, question_answering]
task: Question Answering
language: ar
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `AraELECTRA-discriminator-SOQAL` is a Arabic model originally trained by `Damith`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/electra_qa_AraELECTRA_discriminator_SOQAL_ar_4.0.0_3.0_1655918555995.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/electra_qa_AraELECTRA_discriminator_SOQAL_ar_4.0.0_3.0_1655918555995.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = MultiDocumentAssembler() \
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = BertForQuestionAnswering.pretrained("electra_qa_AraELECTRA_discriminator_SOQAL","ar") \
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

val spanClassifer = BertForQuestionAnswering.pretrained("electra_qa_AraELECTRA_discriminator_SOQAL","ar") 
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
nlu.load("ar.answer_question.electra").predict("""ما هو اسمي؟|||"اسمي كلارا وأنا أعيش في بيركلي.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|electra_qa_AraELECTRA_discriminator_SOQAL|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|ar|
|Size:|504.8 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/Damith/AraELECTRA-discriminator-SOQAL