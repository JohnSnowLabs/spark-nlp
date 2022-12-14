---
layout: model
title: Legal Clauses Multilabel Classifier
author: John Snow Labs
name: legmulticlf_edgar
date: 2022-08-30
tags: [en, legal, classification, clauses, edgar, ledgar, licensed]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
recommended: true
annotator: MultiClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Multilabel Document Classification model, which can be used to identify up to 15 classes in texts. The classes are the following:

- terminations
- assigns
- notices
- amendments
- waivers
- survival
- successors
- governing laws
- severability
- expenses
- assignments
- warranties
- representations
- entire agreements
- counterparts

## Predicted Entities

`terminations`, `assigns`, `notices`, `amendments`, `waivers`, `survival`, `successors`, `governing laws`, `severability`, `expenses`, `assignments`, `warranties`, `representations`, `entire agreements`, `counterparts`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/legal/LEGMULTICLF_LEDGAR/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legmulticlf_edgar_en_1.0.0_3.2_1661858359724.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document = nlp.DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_bert_base_uncased_legal", "en") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")

multiClassifier = nlp.MultiClassifierDLModel.pretrained("legmulticlf_edgar", "en", "legal/models") \
  .setInputCols(["document", "sentence_embeddings"]) \
  .setOutputCol("class")

ledgar_pipeline = Pipeline(
    stages=[document, 
            embeddings,
            multiClassifier])


light_pipeline = LightPipeline(ledgar_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

result = light_pipeline.annotate("""(a) No failure or delay by the Administrative Agent or any Lender in exercising any right or power hereunder shall operate as a waiver thereof, nor shall any single or partial exercise of any such right or power, or any abandonment or discontinuance of steps to enforce such a right or power, preclude any other or further exercise thereof or the exercise of any other right or power. The rights and remedies of the Administrative Agent and the Lenders hereunder are cumulative and are not exclusive of any rights or remedies that they would otherwise have. No waiver of any provision of this Agreement or consent to any departure by the Borrower therefrom shall in any event be effective unless the same shall be permitted by paragraph (b) of this Section, and then such waiver or consent shall be effective only in the specific instance and for the purpose for which given. Without limiting the generality of the foregoing, the making of a Loan shall not be construed as a waiver of any Default, regardless of whether the Administrative Agent or any Lender may have had notice or knowledge of such Default at the time.""")

result["class"]
```

</div>

## Results

```bash
['waivers', 'amendments']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legmulticlf_edgar|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[category]|
|Language:|en|
|Size:|13.9 MB|

## References

Ledgar dataset, available at https://metatext.io/datasets/ledgar, with in-house data augmentation

## Benchmarking

```bash
label              precision    recall    f1-score  support
amendments         0.89         0.66      0.76      2126
expenses           0.74         0.45      0.56       783
assigns            0.82         0.36      0.50      1156
counterparts       0.99         0.97      0.98      1903
entire_agreements  0.98         0.91      0.94      2168
expenses           0.99         0.53      0.70       817
governing_laws     0.96         0.98      0.97      2608
notices            0.94         0.94      0.94      1888
representations    0.91         0.72      0.80       911
severability       0.97         0.95      0.96      1640
successors         0.90         0.50      0.64      1423
survival           0.95         0.85      0.90      1175
terminations       0.62         0.76      0.68       912
waivers            0.92         0.59      0.72      1474
warranties         0.82         0.66      0.73       756
micro-avg          0.92         0.77      0.84     21740
macro-avg          0.89         0.72      0.78     21740
weighted-avg       0.91         0.77      0.82     21740
samples-avg        0.81         0.80      0.80     21740
```
