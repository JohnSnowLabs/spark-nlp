---
layout: model
title: Legal Mechanical Engineering Document Classifier (EURLEX)
author: John Snow Labs
name: legclf_mechanical_engineering_bert
date: 2023-03-06
tags: [en, legal, classification, clauses, mechanical_engineering, licensed, tensorflow]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: LegalClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

European Union (EU) legislation is published in the EUR-Lex portal. All EU laws are annotated by the EU's Publications Office with multiple concepts from the EuroVoc thesaurus, a multilingual thesaurus maintained by the Publications Office.

Given a document, the legclf_mechanical_engineering_bert model, it is a Bert Sentence Embeddings Document Classifier, classifies if the document belongs to the class Mechanical_Engineering or not (Binary Classification) according to EuroVoc labels.

## Predicted Entities

`Mechanical_Engineering`, `Other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_mechanical_engineering_bert_en_1.0.0_3.0_1678111872224.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_mechanical_engineering_bert_en_1.0.0_3.0_1678111872224.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

doc_classifier = legal.ClassifierDLModel.pretrained("legclf_mechanical_engineering_bert", "en", "legal/models")\
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
|[Mechanical_Engineering]|
|[Other]|
|[Other]|
|[Mechanical_Engineering]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_mechanical_engineering_bert|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|21.5 MB|

## References

Train dataset available [here](https://huggingface.co/datasets/lex_glue)

## Benchmarking

```bash

                 label precision recall  f1-score  support
Mechanical_Engineering      0.85   0.93      0.89      125
                 Other      0.92   0.84      0.88      125
              accuracy         -      -      0.88      250
             macro-avg      0.89   0.88      0.88      250
          weighted-avg      0.89   0.88      0.88      250
```