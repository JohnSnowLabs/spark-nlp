---
layout: model
title: Legal Limited Liability Company Operating Agreement Document Classifier (Bert Sentence Embeddings)
author: John Snow Labs
name: legclf_limited_liability_company_operating_agreement_bert
date: 2023-02-02
tags: [en, legal, classification, limited, liability, company, operating, agreement, licensed, bert, tensorflow]
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

The `legclf_limited_liability_company_operating_agreement_bert` model is a Bert Sentence Embeddings Document Classifier used to classify if the document belongs to the class `limited-liability-company-operating-agreement` or not (Binary Classification).

Unlike the Longformer model, this model is lighter in terms of inference time.

## Predicted Entities

`limited-liability-company-operating-agreement`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_limited_liability_company_operating_agreement_bert_en_1.0.0_3.0_1675360839682.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_limited_liability_company_operating_agreement_bert_en_1.0.0_3.0_1675360839682.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
doc_classifier = legal.ClassifierDLModel.pretrained("legclf_limited_liability_company_operating_agreement_bert", "en", "legal/models")\
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
|[limited-liability-company-operating-agreement]|
|[other]|
|[other]|
|[limited-liability-company-operating-agreement]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_limited_liability_company_operating_agreement_bert|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.6 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents 

## Benchmarking

```bash
                                        label  precision    recall  f1-score   support
limited-liability-company-operating-agreement       1.00      0.94      0.97        53
                                        other       0.98      1.00      0.99       122
                                     accuracy          -         -      0.98       175
                                    macro-avg       0.99      0.97      0.98       175
                                 weighted-avg       0.98      0.98      0.98       175
```
