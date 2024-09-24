---
layout: model
title: English sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15 BertForQuestionAnswering from phd411r1
author: John Snow Labs
name: sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15
date: 2024-09-23
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

Pretrained BertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15` is a English model originally trained by phd411r1.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15_en_5.5.0_3.0_1727128026158.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15_en_5.5.0_3.0_1727128026158.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
             
documentAssembler = MultiDocumentAssembler() \
     .setInputCol(["question", "context"]) \
     .setOutputCol(["document_question", "document_context"])
    
spanClassifier = BertForQuestionAnswering.pretrained("sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15","en") \
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
    
val spanClassifier = BertForQuestionAnswering.pretrained("sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15", "en")
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
|Model Name:|sajjadayoubi_bert_base_persian_farsi_qa_finetune_on_amharic_15|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|606.5 MB|

## References

https://huggingface.co/phd411r1/SajjadAyoubi_bert-base-fa-qa_finetune_on_am_15