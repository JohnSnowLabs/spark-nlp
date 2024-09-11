---
layout: model
title: Uzbek uzn_roberta_base_ft_qa_english_maltese_tonga_tonga_islands_uzn RoBertaForQuestionAnswering from med-alex
author: John Snow Labs
name: uzn_roberta_base_ft_qa_english_maltese_tonga_tonga_islands_uzn
date: 2024-09-11
tags: [uz, open_source, onnx, question_answering, roberta]
task: Question Answering
language: uz
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: RoBertaForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`uzn_roberta_base_ft_qa_english_maltese_tonga_tonga_islands_uzn` is a Uzbek model originally trained by med-alex.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/uzn_roberta_base_ft_qa_english_maltese_tonga_tonga_islands_uzn_uz_5.5.0_3.0_1726058330974.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/uzn_roberta_base_ft_qa_english_maltese_tonga_tonga_islands_uzn_uz_5.5.0_3.0_1726058330974.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
             
documentAssembler = MultiDocumentAssembler() \
     .setInputCol(["question", "context"]) \
     .setOutputCol(["document_question", "document_context"])
    
spanClassifier = RoBertaForQuestionAnswering.pretrained("uzn_roberta_base_ft_qa_english_maltese_tonga_tonga_islands_uzn","uz") \
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
    
val spanClassifier = RoBertaForQuestionAnswering.pretrained("uzn_roberta_base_ft_qa_english_maltese_tonga_tonga_islands_uzn", "uz")
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
|Model Name:|uzn_roberta_base_ft_qa_english_maltese_tonga_tonga_islands_uzn|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|uz|
|Size:|311.8 MB|

## References

https://huggingface.co/med-alex/uzn-roberta-base-ft-qa-en-mt-to-uzn