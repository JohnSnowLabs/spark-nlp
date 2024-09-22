---
layout: model
title: English bert_base_uncased_ep_2_2_b_32_lr_1_2e_06_dp_0_3_swati_500_southern_sotho_false_fh_true_hs_0 BertForQuestionAnswering from danielkty22
author: John Snow Labs
name: bert_base_uncased_ep_2_2_b_32_lr_1_2e_06_dp_0_3_swati_500_southern_sotho_false_fh_true_hs_0
date: 2024-09-22
tags: [en, open_source, onnx, question_answering, bert]
task: Question Answering
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_uncased_ep_2_2_b_32_lr_1_2e_06_dp_0_3_swati_500_southern_sotho_false_fh_true_hs_0` is a English model originally trained by danielkty22.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_uncased_ep_2_2_b_32_lr_1_2e_06_dp_0_3_swati_500_southern_sotho_false_fh_true_hs_0_en_5.5.0_3.0_1726992015862.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_uncased_ep_2_2_b_32_lr_1_2e_06_dp_0_3_swati_500_southern_sotho_false_fh_true_hs_0_en_5.5.0_3.0_1726992015862.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
             
documentAssembler = MultiDocumentAssembler() \
     .setInputCol(["question", "context"]) \
     .setOutputCol(["document_question", "document_context"])
    
spanClassifier = BertForQuestionAnswering.pretrained("bert_base_uncased_ep_2_2_b_32_lr_1_2e_06_dp_0_3_swati_500_southern_sotho_false_fh_true_hs_0","en") \
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
    
val spanClassifier = BertForQuestionAnswering.pretrained("bert_base_uncased_ep_2_2_b_32_lr_1_2e_06_dp_0_3_swati_500_southern_sotho_false_fh_true_hs_0", "en")
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
|Model Name:|bert_base_uncased_ep_2_2_b_32_lr_1_2e_06_dp_0_3_swati_500_southern_sotho_false_fh_true_hs_0|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|407.2 MB|

## References

https://huggingface.co/danielkty22/bert-base-uncased-ep-2.2-b-32-lr-1.2e-06-dp-0.3-ss-500-st-False-fh-True-hs-0