---
layout: model
title: Persian XlmRoBertaForQuestionAnswering model (from m3hrdadfi)
author: John Snow Labs
name: xlmroberta_qa_xlmr_large
date: 2022-06-28
tags: [fa, open_source, xlmroberta, question_answering]
task: Question Answering
language: fa
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: XlmRoBertaForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `xlmr-large-qa-fa` is a Persian model originally trained by `m3hrdadfi`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_qa_xlmr_large_fa_4.0.0_3.0_1656419067604.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = MultiDocumentAssembler() \
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = XlmRoBertaForQuestionAnswering.pretrained("xlmroberta_qa_xlmr_large","fa") \
.setInputCols(["document_question", "document_context"]) \
.setOutputCol("answer")\
.setCaseSensitive(True)

pipeline = Pipeline(stages=[documentAssembler, spanClassifier])

data = spark.createDataFrame([["اسم من چیست؟", "نام من کلارا است و من در برکلی زندگی می کنم."]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new MultiDocumentAssembler() 
.setInputCols(Array("question", "context")) 
.setOutputCols(Array("document_question", "document_context"))

val spanClassifer = XlmRoBertaForQuestionAnswering.pretrained("xlmroberta_qa_xlmr_large","fa") 
.setInputCols(Array("document", "token")) 
.setOutputCol("answer")
.setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler, spanClassifier))

val data = Seq("اسم من چیست؟", "نام من کلارا است و من در برکلی زندگی می کنم.").toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("fa.answer_question.xlmr_roberta.large").predict("""اسم من چیست؟|||"نام من کلارا است و من در برکلی زندگی می کنم.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_qa_xlmr_large|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|fa|
|Size:|1.9 GB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/m3hrdadfi/xlmr-large-qa-fa
- https://github.com/sajjjadayobi/PersianQA
- https://github.com/m3hrdadfi