---
layout: model
title: English BertForQuestionAnswering model (from madlag)
author: John Snow Labs
name: bert_qa_bert_base_uncased_squadv1_x2.32_f86.6_d15_hybrid_v1
date: 2022-06-02
tags: [en, open_source, question_answering, bert]
task: Question Answering
language: en
nav_key: models
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-uncased-squadv1-x2.32-f86.6-d15-hybrid-v1` is a English model orginally trained by `madlag`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_qa_bert_base_uncased_squadv1_x2.32_f86.6_d15_hybrid_v1_en_4.0.0_3.0_1654181597581.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_qa_bert_base_uncased_squadv1_x2.32_f86.6_d15_hybrid_v1_en_4.0.0_3.0_1654181597581.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = MultiDocumentAssembler() \ 
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = BertForQuestionAnswering.pretrained("bert_qa_bert_base_uncased_squadv1_x2.32_f86.6_d15_hybrid_v1","en") \
.setInputCols(["document_question", "document_context"]) \
.setOutputCol("answer") \
.setCaseSensitive(True)

pipeline = Pipeline().setStages([
document_assembler,
spanClassifier
])

example = spark.createDataFrame([["What's my name?", "My name is Clara and I live in Berkeley."]]).toDF("question", "context")

result = pipeline.fit(example).transform(example)
```
```scala
val document = new MultiDocumentAssembler()
.setInputCols("question", "context")
.setOutputCols("document_question", "document_context")

val spanClassifier = BertForQuestionAnswering
.pretrained("bert_qa_bert_base_uncased_squadv1_x2.32_f86.6_d15_hybrid_v1","en")
.setInputCols(Array("document_question", "document_context"))
.setOutputCol("answer")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document, spanClassifier))

val example = Seq(
("Where was John Lenon born?", "John Lenon was born in London and lived in Paris. My name is Sarah and I live in London."),
("What's my name?", "My name is Clara and I live in Berkeley."))
.toDF("question", "context")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.answer_question.squad.bert.base_uncased_x2.01_f89.2_d30_hybrid_rewind_opt_v1.by_madlag").predict("""What's my name?|||"My name is Clara and I live in Berkeley.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_qa_bert_base_uncased_squadv1_x2.32_f86.6_d15_hybrid_v1|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|149.1 MB|
|Case sensitive:|false|
|Max sentence length:|512|

## References

- https://huggingface.co/madlag/bert-base-uncased-squadv1-x2.32-f86.6-d15-hybrid-v1
- https://rajpurkar.github.io/SQuAD-explorer
- https://www.aclweb.org/anthology/N19-1423.pdf