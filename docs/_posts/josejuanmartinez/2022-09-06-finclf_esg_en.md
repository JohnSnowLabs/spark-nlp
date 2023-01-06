---
layout: model
title: ESG Text Classification (3 classes)
author: John Snow Labs
name: finclf_esg
date: 2022-09-06
tags: [en, financial, esg, classification, licensed]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model classifies financial texts / news into three classes: Environment, Social and Governance. This model can be use to build a ESG score board for companies.

If you look for an augmented version of this model, with more fine-grain verticals (Green House Emissions, Business Ethics, etc), please look for the finance_sequence_classifier_augmented_esg model in Models Hub.

## Predicted Entities

`Environment`, `Social`, `Governance`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINCLF_ESG/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_esg_en_1.0.0_3.2_1662472406140.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = nlp.Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = finance.BertForSequenceClassification.pretrained("finclf_esg", "en", "finance/models")\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

# couple of simple examples
example = spark.createDataFrame([["""The Canadian Environmental Assessment Agency (CEAA) concluded that in June 2016 the company had not made an effort
 to protect public drinking water and was ignoring concerns raised by its own scientists about the potential levels of pollutants in the local water supply.
  At the time, there were concerns that the company was not fully testing onsite wells for contaminants and did not use the proper methods for testing because 
  of its test kits now manufactured in China.A preliminary report by the company in June 2016 was commissioned by the Alberta government to provide recommendations 
  to Alberta Environment officials"""]]).toDF("text")

result = pipeline.fit(example).transform(example)

# result is a DataFrame
result.select("text", "class.result").show()
```

</div>

## Results

```bash
+--------------------+---------------+
|                text|         result|
+--------------------+---------------+
|The Canadian Envi...|[Environmental]|
+--------------------+---------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_esg|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|412.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

In-house annotations from scrapped annual reports and tweets about ESG

## Benchmarking

```bash
        label   precision    recall  f1-score   support
Environmental        0.99      0.97      0.98        97
       Social        0.95      0.96      0.95       162
   Governance        0.91      0.90      0.91        71
     accuracy           -         -      0.95       330
    macro-avg        0.95      0.94      0.95       330
 weighted-avg        0.95      0.95      0.95       330
```  
