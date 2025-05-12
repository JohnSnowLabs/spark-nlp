---
layout: model
title: English covid_qa_distillbert DistilBertForQuestionAnswering from shainahub
author: John Snow Labs
name: covid_qa_distillbert
date: 2025-04-05
tags: [distilbert, en, open_source, question_answering, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`covid_qa_distillbert` is a English model originally trained by shainahub.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/covid_qa_distillbert_en_5.5.1_3.0_1743890432118.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/covid_qa_distillbert_en_5.5.1_3.0_1743890432118.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = MultiDocumentAssembler() \
    .setInputCol(["question", "context"]) \
    .setOutputCol(["document_question", "document_context"])
    
    
spanClassifier = DistilBertForQuestionAnswering.pretrained("covid_qa_distillbert","en") \
            .setInputCols(["document_question","document_context"]) \
            .setOutputCol("answer")

pipeline = Pipeline().setStages([document_assembler, spanClassifier])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)
```
```scala
val document_assembler = new MultiDocumentAssembler()
    .setInputCol(Array("question", "context")) 
    .setOutputCol(Array("document_question", "document_context"))
    
val spanClassifier = DistilBertForQuestionAnswering  
    .pretrained("covid_qa_distillbert", "en")
    .setInputCols(Array("document_question","document_context")) 
    .setOutputCol("answer") 

val pipeline = new Pipeline().setStages(Array(document_assembler, spanClassifier))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|covid_qa_distillbert|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|247.2 MB|

## References

References

References

https://huggingface.co/shainahub/covid_qa_distillbert