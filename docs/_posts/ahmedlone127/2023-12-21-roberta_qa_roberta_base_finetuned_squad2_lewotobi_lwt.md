---
layout: model
title: Lewotobi roberta_qa_roberta_base_finetuned_squad2_lewotobi RoBertaForQuestionAnswering from 21iridescent
author: John Snow Labs
name: roberta_qa_roberta_base_finetuned_squad2_lewotobi
date: 2023-12-21
tags: [roberta, lwt, open_source, question_answering, onnx]
task: Question Answering
language: lwt
edition: Spark NLP 5.2.1
spark_version: 3.0
supported: true
engine: onnx
annotator: RoBertaForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_qa_roberta_base_finetuned_squad2_lewotobi` is a Lewotobi model originally trained by 21iridescent.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_qa_roberta_base_finetuned_squad2_lewotobi_lwt_5.2.1_3.0_1703193744533.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_qa_roberta_base_finetuned_squad2_lewotobi_lwt_5.2.1_3.0_1703193744533.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = MultiDocumentAssembler() \
    .setInputCol(["question", "context"]) \
    .setOutputCol(["document_question", "document_context"])
    
    
spanClassifier = RoBertaForQuestionAnswering.pretrained("roberta_qa_roberta_base_finetuned_squad2_lewotobi","lwt") \
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
    
val spanClassifier = RoBertaForQuestionAnswering  
    .pretrained("roberta_qa_roberta_base_finetuned_squad2_lewotobi", "lwt")
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
|Model Name:|roberta_qa_roberta_base_finetuned_squad2_lewotobi|
|Compatibility:|Spark NLP 5.2.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|lwt|
|Size:|463.6 MB|

## References

https://huggingface.co/21iridescent/RoBERTa-base-finetuned-squad2-lwt