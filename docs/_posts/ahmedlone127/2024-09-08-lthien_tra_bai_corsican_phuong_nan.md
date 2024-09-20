---
layout: model
title: None lthien_tra_bai_corsican_phuong DistilBertForQuestionAnswering from hi113
author: John Snow Labs
name: lthien_tra_bai_corsican_phuong
date: 2024-09-08
tags: [nan, open_source, onnx, question_answering, distilbert]
task: Question Answering
language: nan
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`lthien_tra_bai_corsican_phuong` is a None model originally trained by hi113.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lthien_tra_bai_corsican_phuong_nan_5.5.0_3.0_1725818443258.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/lthien_tra_bai_corsican_phuong_nan_5.5.0_3.0_1725818443258.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
             
documentAssembler = MultiDocumentAssembler() \
     .setInputCol(["question", "context"]) \
     .setOutputCol(["document_question", "document_context"])
    
spanClassifier = DistilBertForQuestionAnswering.pretrained("lthien_tra_bai_corsican_phuong","nan") \
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
    
val spanClassifier = DistilBertForQuestionAnswering.pretrained("lthien_tra_bai_corsican_phuong", "nan")
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
|Model Name:|lthien_tra_bai_corsican_phuong|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|nan|
|Size:|247.2 MB|

## References

https://huggingface.co/hi113/ltHien_Tra_Bai_Co_Phuong