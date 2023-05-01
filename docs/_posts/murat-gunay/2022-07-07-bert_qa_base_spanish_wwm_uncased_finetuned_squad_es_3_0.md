---
layout: model
title: Spanish BertForQuestionAnswering Base Uncased model (from stevemobs)
author: John Snow Labs
name: bert_qa_base_spanish_wwm_uncased_finetuned_squad
date: 2022-07-07
tags: [es, open_source, bert, question_answering]
task: Question Answering
language: es
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-spanish-wwm-uncased-finetuned-squad_es` is a Spanish model originally trained by `stevemobs`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_qa_base_spanish_wwm_uncased_finetuned_squad_es_4.0.0_3.0_1657183450072.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_qa_base_spanish_wwm_uncased_finetuned_squad_es_4.0.0_3.0_1657183450072.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = MultiDocumentAssembler() \
    .setInputCols(["question", "context"]) \
    .setOutputCols(["document_question", "document_context"])

spanClassifier = BertForQuestionAnswering.pretrained("bert_qa_base_spanish_wwm_uncased_finetuned_squad","es") \
    .setInputCols(["document_question", "document_context"]) \
    .setOutputCol("answer")\
    .setCaseSensitive(True)
    
pipeline = Pipeline(stages=[documentAssembler, spanClassifier])

data = spark.createDataFrame([["¿Cuál es mi nombre?", "Mi nombre es Clara y vivo en Berkeley."]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new MultiDocumentAssembler() 
      .setInputCols(Array("question", "context")) 
      .setOutputCols(Array("document_question", "document_context"))
 
val spanClassifer = BertForQuestionAnswering.pretrained("bert_qa_base_spanish_wwm_uncased_finetuned_squad","es") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("answer")
    .setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(documentAssembler, spanClassifier))

val data = Seq("¿Cuál es mi nombre?", "Mi nombre es Clara y vivo en Berkeley.").toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.answer_question.bert.squad_es.uncased_base_finetuned").predict("""�Cu�l es mi nombre?|||"Mi nombre es Clara y vivo en Berkeley.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_qa_base_spanish_wwm_uncased_finetuned_squad|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|es|
|Size:|410.3 MB|
|Case sensitive:|false|
|Max sentence length:|512|

## References

- https://huggingface.co/stevemobs/bert-base-spanish-wwm-uncased-finetuned-squad_es