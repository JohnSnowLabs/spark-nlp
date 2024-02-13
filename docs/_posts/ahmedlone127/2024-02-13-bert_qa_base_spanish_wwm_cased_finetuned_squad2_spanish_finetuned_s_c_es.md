---
layout: model
title: Castilian, Spanish bert_qa_base_spanish_wwm_cased_finetuned_squad2_spanish_finetuned_s_c BertForQuestionAnswering from MMG
author: John Snow Labs
name: bert_qa_base_spanish_wwm_cased_finetuned_squad2_spanish_finetuned_s_c
date: 2024-02-13
tags: [bert, es, open_source, question_answering, onnx]
task: Question Answering
language: es
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_qa_base_spanish_wwm_cased_finetuned_squad2_spanish_finetuned_s_c` is a Castilian, Spanish model originally trained by MMG.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_qa_base_spanish_wwm_cased_finetuned_squad2_spanish_finetuned_s_c_es_5.2.4_3.0_1707856281553.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_qa_base_spanish_wwm_cased_finetuned_squad2_spanish_finetuned_s_c_es_5.2.4_3.0_1707856281553.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = MultiDocumentAssembler() \
    .setInputCol(["question", "context"]) \
    .setOutputCol(["document_question", "document_context"])
    
    
spanClassifier = BertForQuestionAnswering.pretrained("bert_qa_base_spanish_wwm_cased_finetuned_squad2_spanish_finetuned_s_c","es") \
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
    
val spanClassifier = BertForQuestionAnswering  
    .pretrained("bert_qa_base_spanish_wwm_cased_finetuned_squad2_spanish_finetuned_s_c", "es")
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
|Model Name:|bert_qa_base_spanish_wwm_cased_finetuned_squad2_spanish_finetuned_s_c|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|es|
|Size:|409.5 MB|

## References

References

https://huggingface.co/MMG/bert-base-spanish-wwm-cased-finetuned-squad2-es-finetuned-sqac