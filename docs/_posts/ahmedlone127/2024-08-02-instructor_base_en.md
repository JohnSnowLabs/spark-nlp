---
layout: model
title: Instructor Base Sentence Embeddings
author: John Snow Labs
name: instructor_base
date: 2024-08-02
tags: [en, instructor, sentence_embeddings, text_reranking, open_source, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: InstructorEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Instructor👨‍🏫, an instruction-finetuned text embedding model that can generate text embeddings tailored to any task (e.g., classification, retrieval, clustering, text evaluation, etc.) and domains (e.g., science, finance, etc.) by simply providing the task instruction, without any finetuning. Instructor👨‍ achieves sota on 70 diverse embedding tasks.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/instructor_base_en_5.4.2_3.0_1722602498531.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/instructor_base_en_5.4.2_3.0_1722602498531.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

instruction = InstructorEmbeddings.pretrained("instructor_base","en") \
            .setInstruction("Instruction here: ") \
            .setInputCols(["documents"]) \
            .setOutputCol("instructor")

        pipeline = Pipeline().setStages([document_assembler, instruction])

```
```scala

    val embeddings = InstructorEmbeddings
      .pretrained("instructor_base","en")
      .setInstruction("Instruction here: ")
      .setInputCols(Array("document"))
      .setOutputCol("instructor")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|instructor_base|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[instructor]|
|Language:|en|
|Size:|406.0 MB|