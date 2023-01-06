---
layout: model
title: Financial NER (xl, Extra Large)
author: John Snow Labs
name: finner_financial_xlarge
date: 2022-11-30
tags: [en, financial, ner, earning, calls, 10k, fillings, annual, reports, licensed]
task: Named Entity Recognition
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a `xl` (extra-large) version of a financial model, trained in a combination of two data sets: Earning Calls and 10K Fillings.

Please note this model requires some tokenization configuration to extract the currency (see python snippet below).

The aim of this model is to detect the main pieces of financial information in annual reports of companies, more specifically this model is being trained with 10K filings.

The currently available entities are:
- AMOUNT: Numeric amounts, not percentages
- ASSET: Current or Fixed Asset
- ASSET_DECREASE: Decrease in the asset possession/exposure
- ASSET_INCREASE: Increase in the asset possession/exposure
- CF: Total cash flow 
- CF_DECREASE: Relative decrease in cash flow
- CF_INCREASE: Relative increase in cash flow
- COUNT: Number of items (not monetary, not percentages).
- CURRENCY: The currency of the amount
- DATE: Generic dates in context where either it's not a fiscal year or it can't be asserted as such given the context
- EXPENSE: An expense or loss
- EXPENSE_DECREASE: A piece of information saying there was an expense decrease in that fiscal year
- EXPENSE_INCREASE: A piece of information saying there was an expense increase in that fiscal year
- FCF: Free Cash Flow
- FISCAL_YEAR: A date which expresses which month the fiscal exercise was closed for a specific year
- KPI: Key Performance Indicator, a quantifiable measure of performance over time for a specific objective
- KPI_DECREASE: Relative decrease in a KPI
- KPI_INCREASE: Relative increase in a KPI
- LIABILITY:  Current or Long-Term Liability (not from stockholders)
- LIABILITY_DECREASE: Relative decrease in liability
- LIABILITY_INCREASE: Relative increase in liability
- ORG: Mention to a company/organization name
- PERCENTAGE: Numeric amounts which are percentages
- PROFIT: Profit or also Revenue
- PROFIT_DECLINE: A piece of information saying there was a profit / revenue decrease in that fiscal year
- PROFIT_INCREASE: A piece of information saying there was a profit / revenue increase in that fiscal year
- TICKER: Trading symbol of the company

You can also check for the Relation Extraction model which connects these entities together

## Predicted Entities

`AMOUNT`, `ASSET`, `ASSET_DECREASE`, `ASSET_INCREASE`, `CF`, `CF_DECREASE`, `CF_INCREASE`, `COUNT`, `CURRENCY`, `DATE`, `EXPENSE`, `EXPENSE_DECREASE`, `EXPENSE_INCREASE`, `FCF`, `FISCAL_YEAR`, `KPI`, `KPI_DECREASE`, `KPI_INCREASE`, `LIABILITY`, `LIABILITY_DECREASE`, `LIABILITY_INCREASE`, `ORG`, `PERCENTAGE`, `PROFIT`, `PROFIT_DECLINE`, `PROFIT_INCREASE`, `TICKER`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_financial_xlarge_en_1.0.0_3.0_1669840074362.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")\
    .setContextChars(['.', ',', ';', ':', '!', '?', '*', '-', '(', ')', '”', '’', '$','€'])

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base", "en") \
  .setInputCols("sentence", "token") \
  .setOutputCol("embeddings")\
  .setMaxSentenceLength(512)

ner_model = finance.NerModel.pretrained("finner_financial_xlarge", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = nlp.Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter   
    ])

data = spark.createDataFrame([["""License fees revenue decreased 40 %, or 0.5 million to 0.7 million for the year ended December 31, 2020 compared to 1.2 million for the year ended December 31, 2019"""]]).toDF("text")

model = pipeline.fit(data)

result = model.transform(data)

result.select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata')).alias("cols")) \
               .select(F.expr("cols['0']").alias("text"),
                       F.expr("cols['1']['entity']").alias("label")).show(200, truncate = False)
```

</div>

## Results

```bash
+---------+----------------+----------+
|    token|       ner_label|confidence|
+---------+----------------+----------+
|  License|B-PROFIT_DECLINE|    0.9658|
|     fees|I-PROFIT_DECLINE|    0.7826|
|  revenue|I-PROFIT_DECLINE|    0.8992|
|decreased|               O|       1.0|
|       40|    B-PERCENTAGE|    0.9997|
|        %|               O|       1.0|
|        ,|               O|    0.9997|
|       or|               O|    0.9999|
|      0.5|        B-AMOUNT|    0.9925|
|  million|        I-AMOUNT|    0.9989|
|       to|               O|    0.9996|
|      0.7|        B-AMOUNT|    0.9368|
|  million|        I-AMOUNT|    0.9949|
|      for|               O|    0.9999|
|      the|               O|    0.9944|
|     year|               O|    0.9976|
|    ended|               O|    0.9987|
| December|   B-FISCAL_YEAR|    0.9941|
|       31|   I-FISCAL_YEAR|    0.8955|
|        ,|   I-FISCAL_YEAR|    0.8869|
|     2020|   I-FISCAL_YEAR|    0.9941|
| compared|               O|    0.9999|
|       to|               O|    0.9995|
|      1.2|        B-AMOUNT|    0.9853|
|  million|        I-AMOUNT|    0.9831|
|      for|               O|    0.9999|
|      the|               O|    0.9914|
|     year|               O|    0.9948|
|    ended|               O|    0.9985|
| December|   B-FISCAL_YEAR|    0.9812|
|       31|   I-FISCAL_YEAR|    0.8185|
|        ,|   I-FISCAL_YEAR|    0.8351|
|     2019|   I-FISCAL_YEAR|    0.9541|
+---------+----------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_financial_xlarge|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.2 MB|

## References

In-house annotations on Earning Calls and 10-K Filings combined.

## Benchmarking

```bash
label                 tp    fp   fn   prec        rec         f1         
B-LIABILITY_INCREASE  1     0    0    1.0         1.0         1.0        
I-AMOUNT              915   97   8    0.9041502   0.9913326   0.9457364  
B-COUNT               7     1    2    0.875       0.7777778   0.8235294  
I-LIABILITY_INCREASE  1     0    0    1.0         1.0         1.0        
B-AMOUNT              1304  124  19   0.9131653   0.9856387   0.9480189  
I-KPI                 2     0    7    1.0         0.22222222  0.36363637 
B-DATE                525   32   44   0.94254935  0.9226714   0.9325044  
I-LIABILITY           156   49   97   0.7609756   0.6166008   0.68122274 
I-DATE                343   12   36   0.9661972   0.9050132   0.93460494 
B-CF_DECREASE         6     1    3    0.85714287  0.6666667   0.75       
I-EXPENSE             270   86   74   0.75842696  0.78488374  0.77142864 
I-KPI_INCREASE        0     0    1    0.0         0.0         0.0        
B-LIABILITY           82    30   46   0.73214287  0.640625    0.6833333  
I-CF                  420   97   84   0.8123791   0.8333333   0.82272285 
I-CF_DECREASE         17    3    12   0.85        0.5862069   0.6938776  
I-COUNT               7     0    0    1.0         1.0         1.0        
B-FCF                 5     0    0    1.0         1.0         1.0        
B-PROFIT_INCREASE     54    23   22   0.7012987   0.7105263   0.7058824  
B-KPI_INCREASE        1     0    2    1.0         0.33333334  0.5        
B-EXPENSE             118   42   36   0.7375      0.76623374  0.75159234 
I-CF_INCREASE         43    0    17   1.0         0.71666664  0.8349514  
I-PERCENTAGE          4     6    0    0.4         1.0         0.5714286  
I-PROFIT_DECLINE      39    11   4    0.78        0.90697676  0.8387097  
I-KPI_DECREASE        1     1    0    0.5         1.0         0.6666667  
B-CF_INCREASE         23    0    2    1.0         0.92        0.9583333  
I-PROFIT              228   118  19   0.6589595   0.9230769   0.7689713  
B-CURRENCY            943   42   12   0.9573604   0.98743457  0.972165   
I-PROFIT_INCREASE     80    34   16   0.7017544   0.8333333   0.7619047  
B-CF                  118   32   29   0.7866667   0.8027211   0.7946128  
B-PROFIT              134   55   23   0.7089947   0.85350317  0.7745664  
B-PERCENTAGE          281   17   7    0.942953    0.9756944   0.95904434 
B-TICKER              2     0    0    1.0         1.0         1.0        
I-FISCAL_YEAR         585   17   27   0.9717608   0.9558824   0.9637562  
B-ORG                 2     0    0    1.0         1.0         1.0        
B-PROFIT_DECLINE      22    5    8    0.8148148   0.73333335  0.7719298  
B-EXPENSE_INCREASE    35    7    4    0.8333333   0.8974359   0.86419755 
B-EXPENSE_DECREASE    23    3    4    0.88461536  0.8518519   0.8679245  
B-FISCAL_YEAR         195   6    12   0.9701493   0.942029    0.9558824  
I-EXPENSE_DECREASE    46    9    16   0.8363636   0.7419355   0.78632486 
I-FCF                 10    0    0    1.0         1.0         1.0        
I-EXPENSE_INCREASE    83    13   9    0.8645833   0.90217394  0.88297874 
Macro-average         7134  977  728  0.77496254  0.72599226  0.74967855 
Micro-average         7134  977  728  0.8795463   0.9074027   0.8932574 
```
