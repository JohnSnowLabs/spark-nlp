---
layout: model
title: Finance Capital Call Notices Document Classifier (Bert Sentence Embeddings)
author: John Snow Labs
name: finclf_capital_call_notices
date: 2023-02-16
tags: [en, licensed, finance, capital_calls, classification, tensorflow]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: LegalClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The `finclf_capital_call_notices` model is a Bert Sentence Embeddings Document Classifier used to classify if the document belongs to the class `capital_call_notices` or not (Binary Classification).

## Predicted Entities

`capital_call_notices`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_capital_call_notices_en_1.0.0_3.0_1676590287518.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_capital_call_notices_en_1.0.0_3.0_1676590287518.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
  
embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en")\
    .setInputCols("document")\
    .setOutputCol("sentence_embeddings")
    
doc_classifier = finance.ClassifierDLModel.pretrained("finclf_capital_call_notices", "en", "finance/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")
    
nlpPipeline = nlp.Pipeline(stages=[
    document_assembler, 
    embeddings,
    doc_classifier])
 
df = spark.createDataFrame([["YOUR TEXT HERE"]]).toDF("text")

model = nlpPipeline.fit(df)

result = model.transform(df)
```

</div>

## Results

```bash
+-------+
|result|
+-------+
|[capital_call_notices]|
|[other]|
|[other]|
|[capital_call_notices]|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_capital_call_notices|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.4 MB|

## References

Financial documents and classified in-house + SEC documents

## Benchmarking

```bash
label                 precision  recall  f1-score  support 
capital_call_notices  1.00       1.00    1.00      12      
other                 1.00       1.00    1.00      23      
accuracy              -          -       1.00      35      
macro-avg             1.00       1.00    1.00      35      
weighted-avg          1.00       1.00    1.00      35   
```
