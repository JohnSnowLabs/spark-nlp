---
layout: model
title: English RobertaForQuestionAnswering Large Cased model (from akdeniz27)
author: John Snow Labs
name: roberta_qa_large_cuad
date: 2022-12-02
tags: [en, open_source, roberta, question_answering, tensorflow]
task: Question Answering
language: en
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForQuestionAnswering  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `roberta-large-cuad` is a English model originally trained by `akdeniz27`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_qa_large_cuad_en_4.2.4_3.0_1669987585871.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
Document_Assembler = MultiDocumentAssembler()\
     .setInputCols(["question", "context"])\
     .setOutputCols(["document_question", "document_context"])

Question_Answering = RoBertaForQuestionAnswering.pretrained("roberta_qa_large_cuad","en")\
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

val Question_Answering = RoBertaForQuestionAnswering.pretrained("roberta_qa_large_cuad","en")
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
|Model Name:|roberta_qa_large_cuad|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/akdeniz27/roberta-large-cuad
- https://www.atticusprojectai.org/cuad
- https://github.com/TheAtticusProject/cuad
- https://arxiv.org/abs/2103.06268
- https://aclanthology.org/2021.acl-long.330.pdf
- https://dl.acm.org/doi/pdf/10.1145/3442188.3445922
- https://drive.google.com/file/d/1of37X0hAhECQ3BN_004D8gm6V88tgZaB/view?usp=sharing
- https://zenodo.org/record/4599830
- https://mlco2.github.io/impact#compute
- https://arxiv.org/abs/1910.09700
- https://www.atticusprojectai.org/cuad
- https://www.atticusprojectai.org/