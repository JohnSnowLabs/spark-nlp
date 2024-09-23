---
layout: model
title: English qa_synthetic_data_only_finetuned_v1_0 XlmRoBertaForQuestionAnswering from am-infoweb
author: John Snow Labs
name: qa_synthetic_data_only_finetuned_v1_0
date: 2024-09-05
tags: [en, open_source, onnx, question_answering, xlm_roberta]
task: Question Answering
language: en
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

Pretrained XlmRoBertaForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`qa_synthetic_data_only_finetuned_v1_0` is a English model originally trained by am-infoweb.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/qa_synthetic_data_only_finetuned_v1_0_en_5.5.0_3.0_1725557042495.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/qa_synthetic_data_only_finetuned_v1_0_en_5.5.0_3.0_1725557042495.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
             
documentAssembler = MultiDocumentAssembler() \
     .setInputCol(["question", "context"]) \
     .setOutputCol(["document_question", "document_context"])
    
spanClassifier = XlmRoBertaForQuestionAnswering.pretrained("qa_synthetic_data_only_finetuned_v1_0","en") \
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
    
val spanClassifier = XlmRoBertaForQuestionAnswering.pretrained("qa_synthetic_data_only_finetuned_v1_0", "en")
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
|Model Name:|qa_synthetic_data_only_finetuned_v1_0|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|803.6 MB|

## References

https://huggingface.co/am-infoweb/QA_SYNTHETIC_DATA_ONLY_Finetuned_v1.0