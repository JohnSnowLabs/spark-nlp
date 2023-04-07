---
layout: model
title: Portuguese BertForQuestionAnswering Base Cased model (from pierreguillou)
author: John Snow Labs
name: Bert_qa_base_cased_squad_v1.1_portuguese
date: 2023-04-05
tags: [pt, open_source, bert, question_answering, tensorflow]
task: Question Answering
language: pt
edition: Spark NLP 4.4.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-cased-squad-v1.1-portuguese` is a Portuguese model originally trained by `pierreguillou`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/Bert_qa_base_cased_squad_v1.1_portuguese_pt_4.4.0_3.0_1680696828024.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/Bert_qa_base_cased_squad_v1.1_portuguese_pt_4.4.0_3.0_1680696828024.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
Document_Assembler = MultiDocumentAssembler()\
     .setInputCols(["question", "context"])\
     .setOutputCols(["document_question", "document_context"])

Question_Answering = BertForQuestionAnswering.pretrained("Bert_qa_base_cased_squad_v1.1_portuguese","pt")\
     .setInputCols(["document_question", "document_context"])\
     .setOutputCol("answer")\
     .setCaseSensitive(True)
    
pipeline = Pipeline(stages=[Document_Assembler, Question_Answering])

data = spark.createDataFrame([["What's my name?","My name is Clara and I live in Berkeley."]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)
```
```scala
val Document_Assembler = new MultiDocumentAssembler()
     .setInputCols(Array("question", "context"))
     .setOutputCols(Array("document_question", "document_context"))

val Question_Answering = BertForQuestionAnswering.pretrained("Bert_qa_base_cased_squad_v1.1_portuguese","pt")
     .setInputCols(Array("document_question", "document_context"))
     .setOutputCol("answer")
     .setCaseSensitive(true)
    
val pipeline = new Pipeline().setStages(Array(Document_Assembler, Question_Answering))

val data = Seq("What's my name?","My name is Clara and I live in Berkeley.").toDS.toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|Bert_qa_base_cased_squad_v1.1_portuguese|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|pt|
|Size:|406.5 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

- https://huggingface.co/pierreguillou/bert-base-cased-squad-v1.1-portuguese
- https://miro.medium.com/max/2000/1*te5MmdesAHCmg4KmK8zD3g.png
- http://www.deeplearningbrasil.com.br/
- https://neuralmind.ai/
- https://medium.com/@pierre_guillou/nlp-modelo-de-question-answering-em-qualquer-idioma-baseado-no-bert-base-estudo-de-caso-em-12093d385e78
- https://colab.research.google.com/drive/18ueLdi_V321Gz37x4gHq8mb4XZSGWfZx?usp=sharing
- https://github.com/piegu/language-models/blob/master/colab_question_answering_BERT_base_cased_squad_v11_pt.ipynb
- https://www.linkedin.com/in/pierreguillou/
- https://medium.com/@pierre_guillou/nlp-modelo-de-question-answering-em-qualquer-idioma-baseado-no-bert-base-estudo-de-caso-em-12093d385e78#c572
- https://neuralmind.ai/
- http://www.deeplearningbrasil.com.br/
- https://colab.research.google.com/
- https://ailab.unb.br/