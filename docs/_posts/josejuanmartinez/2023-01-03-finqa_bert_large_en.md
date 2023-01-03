---
layout: model
title: Financial Question Answering (Bert, Large)
author: John Snow Labs
name: finqa_bert_large
date: 2023-01-03
tags: [en, licensed]
task: Question Answering
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Financial large Bert-based Question Answering model, trained on squad-v2, finetuned on proprietary Financial questions and answers.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finqa_bert_large_en_1.0.0_3.0_1672759452867.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.MultiDocumentAssembler()\
        .setInputCols(["question", "context"])\
        .setOutputCols(["document_question", "document_context"])

spanClassifier = nlp.BertForQuestionAnswering.pretrained("finqa_bert_large","en", "finance/models") \
       .setInputCols(["document_question", "document_context"]) \
       .setOutputCol("answer") \
       .setCaseSensitive(True)

pipeline = Pipeline().setStages([
        documentAssembler,
        spanClassifier
])

example = spark.createDataFrame([["On which market is their common stock traded?", "Our common stock is traded on the Nasdaq Global Select Market under the symbol CDNS."]]).toDF("question", "context")

result = pipeline.fit(example).transform(example)
result.select('answer.result').show()
```

</div>

## Results

```bash
`Nasdaq Global Select Market under the symbol CDNS`
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finqa_bert_large|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Trained on squad-v2, finetuned on proprietary Financial questions and answers.