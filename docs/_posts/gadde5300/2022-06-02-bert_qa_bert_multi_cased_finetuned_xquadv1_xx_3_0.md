---
layout: model
title: Multilingual BertForQuestionAnswering model (from mrm8488)
author: John Snow Labs
name: bert_qa_bert_multi_cased_finetuned_xquadv1
date: 2022-06-02
tags: [en, es, de, el, ru, tr, ar, vi, th, zh, hi, open_source, question_answering, bert, xx]
task: Question Answering
language: xx
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertForQuestionAnswering
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Question Answering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-multi-cased-finetuned-xquadv1` is a Multilingual model orginally trained by `mrm8488`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_qa_bert_multi_cased_finetuned_xquadv1_xx_4.0.0_3.0_1654184515717.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = MultiDocumentAssembler() \ 
.setInputCols(["question", "context"]) \
.setOutputCols(["document_question", "document_context"])

spanClassifier = BertForQuestionAnswering.pretrained("bert_qa_bert_multi_cased_finetuned_xquadv1","xx") \
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
.pretrained("bert_qa_bert_multi_cased_finetuned_xquadv1","xx")
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
nlu.load("xx.answer_question.xquad.bert.cased").predict("""What's my name?|||"My name is Clara and I live in Berkeley.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_qa_bert_multi_cased_finetuned_xquadv1|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|xx|
|Size:|665.6 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/mrm8488/bert-multi-cased-finetuned-xquadv1
- https://github.com/google-research/bert/blob/master/multilingual.md
- https://twitter.com/mrm8488
- https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl
- https://colab.research.google.com/github/mrm8488/shared_colab_notebooks/blob/master/Try_mrm8488_xquad_finetuned_model.ipynb
- https://github.com/fxsjy/jieba
- https://github.com/deepmind/xquad