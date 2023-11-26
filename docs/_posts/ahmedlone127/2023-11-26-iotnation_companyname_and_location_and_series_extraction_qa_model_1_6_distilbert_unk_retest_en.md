---
layout: model
title: English iotnation_companyname_and_location_and_series_extraction_qa_model_1_6_distilbert_unk_retest DistilBertForQuestionAnswering from chriskim2273
author: John Snow Labs
name: iotnation_companyname_and_location_and_series_extraction_qa_model_1_6_distilbert_unk_retest
date: 2023-11-26
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

Pretrained DistilBertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`iotnation_companyname_and_location_and_series_extraction_qa_model_1_6_distilbert_unk_retest` is a English model originally trained by chriskim2273.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/iotnation_companyname_and_location_and_series_extraction_qa_model_1_6_distilbert_unk_retest_en_5.2.0_3.0_1701015865776.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/iotnation_companyname_and_location_and_series_extraction_qa_model_1_6_distilbert_unk_retest_en_5.2.0_3.0_1701015865776.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = MultiDocumentAssembler() \
    .setInputCol(["question", "context"]) \
    .setOutputCol(["document_question", "document_context"])
    
    
spanClassifier = DistilBertForQuestionAnswering.pretrained("iotnation_companyname_and_location_and_series_extraction_qa_model_1_6_distilbert_unk_retest","en") \
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
    .pretrained("iotnation_companyname_and_location_and_series_extraction_qa_model_1_6_distilbert_unk_retest", "en")
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
|Model Name:|iotnation_companyname_and_location_and_series_extraction_qa_model_1_6_distilbert_unk_retest|
|Compatibility:|Spark NLP 5.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|247.2 MB|

## References

https://huggingface.co/chriskim2273/IOTNation_CompanyName_AND_Location_AND_Series_Extraction_QA_Model_1.6_DistilBert_UNK_RETEST