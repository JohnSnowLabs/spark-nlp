---
layout: model
title: Zero-shot Legal NER (CUAD, small)
author: John Snow Labs
name: legner_roberta_zeroshot_cuad_small
date: 2023-01-30
tags: [zero, shot, cuad, en, licensed, tensorflow]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: RoBertaForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Zero-shot NER model, trained using Roberta on SQUAD and finetuned to perform Zero-shot NER using CUAD legal dataset. In order to use it, a specific prompt is required. This is an example of it for extracting PARTIES:

```
"Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract"
```

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_roberta_zeroshot_cuad_small_en_1.0.0_3.0_1675089181024.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_roberta_zeroshot_cuad_small_en_1.0.0_3.0_1675089181024.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")

zeroshot = nlp.ZeroShotNerModel.pretrained("legner_roberta_zeroshot_cuad_small","en","legal/models")\
    .setInputCols(["document", "token"])\
    .setOutputCol("zero_shot_ner")\
    .setEntityDefinitions(
        {
            'PARTIES': ['Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract']
        })

nerconverter = NerConverter()\
  .setInputCols(["document", "token", "zero_shot_ner"])\
  .setOutputCol("ner_chunk")


pipeline = nlp.Pipeline().setStages([
    document_assembler,
    tokenizer,
    zeroshot,
    nerconverter
])

from pyspark.sql import types as T
sample_text = ["""THIS CREDIT AGREEMENT is dated as of April 29, 2010, and is made by and
        among P.H. GLATFELTER COMPANY, a Pennsylvania corporation ( the "COMPANY") and
        certain of its subsidiaries. Identified on the signature pages hereto (each a
        "BORROWER" and collectively, the "BORROWERS"), each of the GUARANTORS (as
        hereinafter defined), the LENDERS (as hereinafter defined), PNC BANK, NATIONAL
        ASSOCIATION, in its capacity as agent for the Lenders under this Agreement
        (hereinafter referred to in such capacity as the "ADMINISTRATIVE AGENT"), and,
        for the limited purpose of public identification in trade tables, PNC CAPITAL
        MARKETS LLC and CITIZENS BANK OF PENNSYLVANIA, as joint arrangers and joint
        bookrunners, and CITIZENS BANK OF PENNSYLVANIA, as syndication agent.""".replace('\n',' ')]

p_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

res = p_model.transform(spark.createDataFrame(sample_text, T.StringType()).toDF("text"))

res.show()
```

</div>

## Results

```bash
+-----------------------+---------+
|chunk                  |ner_label|
+-----------------------+---------+
|P.H. GLATFELTER COMPANY|PARTIES  |
+-----------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_roberta_zeroshot_cuad_small|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|449.0 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

SQUAD and CUAD