---
layout: model
title: Russian distilbert_qa_model_qa_5_epoch_russian DistilBertForQuestionAnswering from AndrewChar
author: John Snow Labs
name: distilbert_qa_model_qa_5_epoch_russian
date: 2024-05-24
tags: [ru, open_source, onnx, question_answering, distilbert]
task: Question Answering
language: ru
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_qa_model_qa_5_epoch_russian` is a Russian model originally trained by AndrewChar.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_qa_model_qa_5_epoch_russian_ru_5.2.4_3.0_1716563604534.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_qa_model_qa_5_epoch_russian_ru_5.2.4_3.0_1716563604534.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
             
documentAssembler = MultiDocumentAssembler() \
     .setInputCol(["question", "context"]) \
     .setOutputCol(["document_question", "document_context"])
    
spanClassifier = DistilBertForQuestionAnswering.pretrained("distilbert_qa_model_qa_5_epoch_russian","ru") \
     .setInputCols(["document_question","document_context"]) \
     .setOutputCol("answer")

pipeline = Pipeline().setStages([documentAssembler, spanClassifier])
data = spark.createDataFrame([["What framework do I use?","I use spark-nlp."]]).toDF("document_question", "document_context")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new MultiDocumentAssembler()
    .setInputCol(Array("question", "context")) 
    .setOutputCol(Array("document_question", "document_context"))
    
val spanClassifier = DistilBertForQuestionAnswering.pretrained("distilbert_qa_model_qa_5_epoch_russian", "ru")
    .setInputCols(Array("document_question","document_context")) 
    .setOutputCol("answer") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, spanClassifier))
val data = Seq("What framework do I use?","I use spark-nlp.").toDS.toDF("document_question", "document_context")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_qa_model_qa_5_epoch_russian|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|ru|
|Size:|505.4 MB|

## References

https://huggingface.co/AndrewChar/model-QA-5-epoch-RU