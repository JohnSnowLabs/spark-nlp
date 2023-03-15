---
layout: model
title: Legal Question Answering (RoBerta, CUAD, Base)
author: John Snow Labs
name: legqa_roberta_cuad_base
date: 2023-01-30
tags: [en, licensed, tensorflow]
task: Question Answering
language: en
nav_key: models
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

Legal RoBerta-based Question Answering model, trained on squad-v2, finetuned on CUAD dataset (base). In order to use it, a specific prompt is required. This is an example of it for extracting PARTIES:

```
"Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract"
```

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legqa_roberta_cuad_base_en_1.0.0_3.0_1675083334950.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legqa_roberta_cuad_base_en_1.0.0_3.0_1675083334950.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

documentAssembler = nlp.MultiDocumentAssembler()\
        .setInputCols(["question", "context"])\
        .setOutputCols(["document_question", "document_context"])

spanClassifier = nlp.RoBertaForQuestionAnswering.pretrained("legqa_roberta_cuad_base","en", "legal/models") \
.setInputCols(["document_question", "document_context"]) \
.setOutputCol("answer") \
.setCaseSensitive(True)

pipeline = nlp.Pipeline().setStages([
documentAssembler,
spanClassifier
])

text = """THIS CREDIT AGREEMENT is dated as of April 29, 2010, and is made by and
      among P.H. GLATFELTER COMPANY, a Pennsylvania corporation ( the "COMPANY") and
      certain of its subsidiaries. Identified on the signature pages hereto (each a
      "BORROWER" and collectively, the "BORROWERS"), each of the GUARANTORS (as
      hereinafter defined), the LENDERS (as hereinafter defined), PNC BANK, NATIONAL
      ASSOCIATION, in its capacity as agent for the Lenders under this Agreement
      (hereinafter referred to in such capacity as the "ADMINISTRATIVE AGENT"), and,
      for the limited purpose of public identification in trade tables, PNC CAPITAL
      MARKETS LLC and CITIZENS BANK OF PENNSYLVANIA, as joint arrangers and joint
      bookrunners, and CITIZENS BANK OF PENNSYLVANIA, as syndication agent.""".replace('\n',' ')
        
        
question = ['"Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract"']

qt = [ [q,text] for q in questions    ]

example = spark.createDataFrame(qt).toDF("question", "context")

result = pipeline.fit(example).transform(example)

result.select('document_question.result', 'answer.result').show(truncate=False)
```

</div>

## Results

```bash
["Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract"]|[P . H . GLATFELTER COMPANY]|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legqa_roberta_cuad_base|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|453.7 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

Squad, finetuned with CUAD-based Question/Answering
