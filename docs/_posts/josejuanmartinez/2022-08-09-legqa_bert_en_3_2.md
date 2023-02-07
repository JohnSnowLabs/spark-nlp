---
layout: model
title: Legal Question Answering (Bert)
author: John Snow Labs
name: legqa_bert
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legqa_bert_en_1.0.0_3.2_1660054695560.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legqa_bert_en_1.0.0_3.2_1660054695560.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.MultiDocumentAssembler()\
        .setInputCols(["question", "context"])\
        .setOutputCols(["document_question", "document_context"])

spanClassifier = nlp.BertForQuestionAnswering.pretrained("legqa_bert","en", "legal/models") \
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
|Model Name:|legqa_bert|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|407.9 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Trained on squad-v2, finetuned on proprietary Legal questions and answers.