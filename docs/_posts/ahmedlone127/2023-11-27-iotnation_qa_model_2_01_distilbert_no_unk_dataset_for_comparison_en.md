---
layout: model
title: English iotnation_qa_model_2_01_distilbert_no_unk_dataset_for_comparison DistilBertForQuestionAnswering from chriskim2273
author: John Snow Labs
name: iotnation_qa_model_2_01_distilbert_no_unk_dataset_for_comparison
date: 2023-11-27
tags: [distilbert, en, open_source, question_answering, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.2.0
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`iotnation_qa_model_2_01_distilbert_no_unk_dataset_for_comparison` is a English model originally trained by chriskim2273.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/iotnation_qa_model_2_01_distilbert_no_unk_dataset_for_comparison_en_5.2.0_3.0_1701073055473.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/iotnation_qa_model_2_01_distilbert_no_unk_dataset_for_comparison_en_5.2.0_3.0_1701073055473.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = MultiDocumentAssembler() \
    .setInputCol(["question", "context"]) \
    .setOutputCol(["document_question", "document_context"])
    
    
spanClassifier = DistilBertForQuestionAnswering.pretrained("iotnation_qa_model_2_01_distilbert_no_unk_dataset_for_comparison","en") \
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
    .pretrained("iotnation_qa_model_2_01_distilbert_no_unk_dataset_for_comparison", "en")
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
|Model Name:|iotnation_qa_model_2_01_distilbert_no_unk_dataset_for_comparison|
|Compatibility:|Spark NLP 5.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|247.2 MB|

## References

https://huggingface.co/chriskim2273/IOTNation_QA_Model_2.01_DistilBert_NO_UNK_DATASET_FOR_COMPARISON