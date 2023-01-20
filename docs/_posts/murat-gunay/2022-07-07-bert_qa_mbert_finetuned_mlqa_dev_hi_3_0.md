---
layout: model
title: Hindi BertForQuestionAnswering Cased model (from roshnir)
author: John Snow Labs
name: bert_qa_mbert_finetuned_mlqa_dev
date: 2022-07-07
tags: [hi, open_source, bert, question_answering]
task: Question Answering
language: hi
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `mBert-finetuned-mlqa-dev-hi` is a Hindi model originally trained by `roshnir`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_qa_mbert_finetuned_mlqa_dev_hi_4.0.0_3.0_1657190202881.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_qa_mbert_finetuned_mlqa_dev_hi_4.0.0_3.0_1657190202881.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = MultiDocumentAssembler() \
    .setInputCols(["question", "context"]) \
    .setOutputCols(["document_question", "document_context"])

spanClassifier = BertForQuestionAnswering.pretrained("bert_qa_mbert_finetuned_mlqa_dev","hi") \
    .setInputCols(["document_question", "document_context"]) \
    .setOutputCol("answer")\
    .setCaseSensitive(True)
    
pipeline = Pipeline(stages=[documentAssembler, spanClassifier])

data = spark.createDataFrame([["मेरा नाम क्या है?", "मेरा नाम क्लारा है और मैं बर्कले में रहता हूं।"]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new MultiDocumentAssembler() 
      .setInputCols(Array("question", "context")) 
      .setOutputCols(Array("document_question", "document_context"))
 
val spanClassifer = BertForQuestionAnswering.pretrained("bert_qa_mbert_finetuned_mlqa_dev","hi") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("answer")
    .setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler, spanClassifier))

val data = Seq("मेरा नाम क्या है?", "मेरा नाम क्लारा है और मैं बर्कले में रहता हूं।").toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_qa_mbert_finetuned_mlqa_dev|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|hi|
|Size:|626.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/roshnir/mBert-finetuned-mlqa-dev-hi