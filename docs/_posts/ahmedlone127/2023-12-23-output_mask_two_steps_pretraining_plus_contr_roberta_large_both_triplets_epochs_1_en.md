---
layout: model
title: English output_mask_two_steps_pretraining_plus_contr_roberta_large_both_triplets_epochs_1 RoBertaForQuestionAnswering from AnonymousSub
author: John Snow Labs
name: output_mask_two_steps_pretraining_plus_contr_roberta_large_both_triplets_epochs_1
date: 2023-12-23
tags: [roberta, en, open_source, question_answering, onnx]
task: Question Answering
language: en
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

Pretrained RoBertaForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`output_mask_two_steps_pretraining_plus_contr_roberta_large_both_triplets_epochs_1` is a English model originally trained by AnonymousSub.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/output_mask_two_steps_pretraining_plus_contr_roberta_large_both_triplets_epochs_1_en_5.2.1_3.0_1703328999827.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/output_mask_two_steps_pretraining_plus_contr_roberta_large_both_triplets_epochs_1_en_5.2.1_3.0_1703328999827.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = MultiDocumentAssembler() \
    .setInputCol(["question", "context"]) \
    .setOutputCol(["document_question", "document_context"])
    
    
spanClassifier = RoBertaForQuestionAnswering.pretrained("output_mask_two_steps_pretraining_plus_contr_roberta_large_both_triplets_epochs_1","en") \
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
    .pretrained("output_mask_two_steps_pretraining_plus_contr_roberta_large_both_triplets_epochs_1", "en")
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
|Model Name:|output_mask_two_steps_pretraining_plus_contr_roberta_large_both_triplets_epochs_1|
|Compatibility:|Spark NLP 5.2.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/AnonymousSub/output_mask_TWO_STEPS_pretraining_plus_contr_roberta-large_BOTH_TRIPLETS_EPOCHS_1