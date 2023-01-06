---
layout: model
title: ESG Text Classification (Augmented, 26 classes)
author: John Snow Labs
name: finclf_augmented_esg
date: 2022-09-06
tags: [en, financial, esg, classification, licensed]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
recommended: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model classifies financial texts / news into 26 ESG classes which belong to three verticals: Environment, Social and Governance. This model can be use to build a ESG score board for companies.

If you look for generic version, only returning Environment, Social or Governance, please look for the finance_sequence_classifier_esg model in Models Hub.

## Predicted Entities

`Business_Ethics`, `Data_Security`, `Access_And_Affordability`, `Business_Model_Resilience`, `Competitive_Behavior`, `Critical_Incident_Risk_Management`, `Customer_Welfare`, `Director_Removal`, `Employee_Engagement_Inclusion_And_Diversity`, `Employee_Health_And_Safety`, `Human_Rights_And_Community_Relations`, `Labor_Practices`, `Management_Of_Legal_And_Regulatory_Framework`, `Physical_Impacts_Of_Climate_Change`, `Product_Quality_And_Safety`, `Product_Design_And_Lifecycle_Management`, `Selling_Practices_And_Product_Labeling`, `Supply_Chain_Management`, `Systemic_Risk_Management`, `Waste_And_Hazardous_Materials_Management`, `Water_And_Wastewater_Management`, `Air_Quality`, `Customer_Privacy`, `Ecological_Impacts`, `Energy_Management`, `GHG_Emissions`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINCLF_ESG/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_augmented_esg_en_1.0.0_3.2_1662473372920.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = finance.BertForSequenceClassification.pretrained("finclf_augmented_esg", "en", "finance/models")\
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
+--------------------+--------------------+
|                text|              result|
+--------------------+--------------------+
|The Canadian Envi...|[Waste_And_Hazard...|
+--------------------+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_augmented_esg|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|410.4 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

In-house annotations from scrapped annual reports and tweets about ESG

## Benchmarking

```bash
label                                             precision     recall       f1-score      support
Business_Ethics                                   0.73          0.80         0.76          10
Data_Security                                     1.00          0.89         0.94           9
Access_And_Affordability                          1.00          1.00         1.00          15
Business_Model_Resilience                         1.00          1.00         1.00          12
Competitive_Behavior                              0.92          1.00         0.96          12
Critical_Incident_Risk_Management                 0.92          1.00         0.96          11
Customer_Welfare                                  0.85          1.00         0.92          11
Director_Removal                                  0.91          1.00         0.95          10
Employee_Engagement_Inclusion_And_Diversity       1.00          1.00         1.00          11
Employee_Health_And_Safety                        1.00          1.00         1.00          10
Human_Rights_And_Community_Relations              0.94          1.00         0.97          16
Labor_Practices                                   0.71          0.53         0.61          19
Management_Of_Legal_And_Regulatory_Framework      1.00          0.95         0.97          19
Physical_Impacts_Of_Climate_Change                0.93          1.00         0.97          14
Product_Quality_And_Safety                        1.00          1.00         1.00          14
Product_Design_And_Lifecycle_Management           1.00          1.00         1.00          18
Selling_Practices_And_Product_Labeling            1.00          1.00         1.00          17
Supply_Chain_Management                           0.89          1.00         0.94           8
Systemic_Risk_Management                          1.00          0.86         0.92          14
Waste_And_Hazardous_Materials_Management          0.88          1.00         0.93          14
Water_And_Wastewater_Management                   1.00          1.00         1.00           8
Air_Quality                                       1.00          1.00         1.00          16
Customer_Privacy                                  1.00          0.93         0.97          15
Ecological_Impacts                                1.00          1.00         1.00          16
Energy_Management                                 1.00          0.91         0.95          11
GHG_Emissions                                     1.00          0.91         0.95          11
accuracy                                             -             -         0.95         330
macro-avg                                         0.95          0.95         0.95         330
weighted-avg                                      0.95          0.95         0.95         330
```
