---
layout: model
title: Legal Question Answering (Bert, Large)
author: John Snow Labs
name: legqa_bert_large
date: 2022-08-09
tags: [en, legal, qa, licensed]
task: Question Answering
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Legal Bert-based Question Answering model, trained on squad-v2, finetuned on proprietary Legal questions and answers.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legqa_bert_large_en_1.0.0_3.2_1660053509660.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.MultiDocumentAssembler()\
        .setInputCols(["question", "context"])\
        .setOutputCols(["document_question", "document_context"])

spanClassifier = nlp.BertForQuestionAnswering.pretrained("legqa_bert_large","en", "legal/models") \
.setInputCols(["document_question", "document_context"]) \
.setOutputCol("answer") \
.setCaseSensitive(True)

pipeline = Pipeline().setStages([
documentAssembler,
spanClassifier
])

example = spark.createDataFrame([["Who was subjected to torture?", "The applicant submitted that her husband was subjected to treatment amounting to abuse whilst in the custody of police."]]).toDF("question", "context")

result = pipeline.fit(example).transform(example)

result.select('answer.result').show()
```

</div>

## Results

```bash
`her husband`
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legqa_bert_large|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Trained on squad-v2, finetuned on proprietary Legal questions and answers.