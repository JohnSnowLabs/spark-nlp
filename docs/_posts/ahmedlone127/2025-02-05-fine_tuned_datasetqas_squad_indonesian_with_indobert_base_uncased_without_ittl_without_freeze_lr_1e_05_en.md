---
layout: model
title: English fine_tuned_datasetqas_squad_indonesian_with_indobert_base_uncased_without_ittl_without_freeze_lr_1e_05 BertForQuestionAnswering from afaji
author: John Snow Labs
name: fine_tuned_datasetqas_squad_indonesian_with_indobert_base_uncased_without_ittl_without_freeze_lr_1e_05
date: 2025-02-05
tags: [en, open_source, onnx, question_answering, bert]
task: Question Answering
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fine_tuned_datasetqas_squad_indonesian_with_indobert_base_uncased_without_ittl_without_freeze_lr_1e_05` is a English model originally trained by afaji.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fine_tuned_datasetqas_squad_indonesian_with_indobert_base_uncased_without_ittl_without_freeze_lr_1e_05_en_5.5.1_3.0_1738763695801.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fine_tuned_datasetqas_squad_indonesian_with_indobert_base_uncased_without_ittl_without_freeze_lr_1e_05_en_5.5.1_3.0_1738763695801.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
             
documentAssembler = MultiDocumentAssembler() \
     .setInputCol(["question", "context"]) \
     .setOutputCol(["document_question", "document_context"])
    
spanClassifier = BertForQuestionAnswering.pretrained("fine_tuned_datasetqas_squad_indonesian_with_indobert_base_uncased_without_ittl_without_freeze_lr_1e_05","en") \
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
    
val spanClassifier = BertForQuestionAnswering.pretrained("fine_tuned_datasetqas_squad_indonesian_with_indobert_base_uncased_without_ittl_without_freeze_lr_1e_05", "en")
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
|Model Name:|fine_tuned_datasetqas_squad_indonesian_with_indobert_base_uncased_without_ittl_without_freeze_lr_1e_05|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|411.7 MB|

## References

https://huggingface.co/afaji/fine-tuned-DatasetQAS-Squad-ID-with-indobert-base-uncased-without-ITTL-without-freeze-LR-1e-05