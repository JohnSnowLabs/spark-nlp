---
layout: model
title: English multi_qa_mpnet_base_dot_v1_finetuned_squad2_all MPNetForQuestionAnswering from haddadalwi
author: John Snow Labs
name: multi_qa_mpnet_base_dot_v1_finetuned_squad2_all
date: 2024-08-31
tags: [en, open_source, onnx, question_answering, mpnet]
task: Question Answering
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: MPNetForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multi_qa_mpnet_base_dot_v1_finetuned_squad2_all` is a English model originally trained by haddadalwi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multi_qa_mpnet_base_dot_v1_finetuned_squad2_all_en_5.4.2_3.0_1725115526541.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multi_qa_mpnet_base_dot_v1_finetuned_squad2_all_en_5.4.2_3.0_1725115526541.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
             
documentAssembler = MultiDocumentAssembler() \
     .setInputCol(["question", "context"]) \
     .setOutputCol(["document_question", "document_context"])
    
spanClassifier = MPNetForQuestionAnswering.pretrained("multi_qa_mpnet_base_dot_v1_finetuned_squad2_all","en") \
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
    
val spanClassifier = MPNetForQuestionAnswering.pretrained("multi_qa_mpnet_base_dot_v1_finetuned_squad2_all", "en")
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
|Model Name:|multi_qa_mpnet_base_dot_v1_finetuned_squad2_all|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|407.0 MB|

## References

https://huggingface.co/haddadalwi/multi-qa-mpnet-base-dot-v1-finetuned-squad2-all