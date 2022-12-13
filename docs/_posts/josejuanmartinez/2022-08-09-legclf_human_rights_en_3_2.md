---
layout: model
title: Human Rights Articles Classification
author: John Snow Labs
name: legclf_human_rights
date: 2022-08-09
tags: [es, legal, conventions, classification, en, licensed]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Roberta-based Legal Sequence Classifier NLP model to label texts about Human Rights in Spanish as one of the following categories:

- `Artículo 1. Obligación de Respetar los Derechos`
- `Artículo 2. Deber de Adoptar Disposiciones de Derecho Interno`
- `Artículo 3. Derecho al Reconocimiento de la Personalidad Jurídica`
- `Artículo 4. Derecho a la Vida`
- `Artículo 5. Derecho a la Integridad Personal`
- `Artículo 6. Prohibición de la Esclavitud y Servidumbre`
- `Artículo 7. Derecho a la Libertad Personal`
- `Artículo 8. Garantías Judiciales`
- `Artículo 9. Principio de Legalidad y de Retroactividad`
- `Artículo 11. Protección de la Honra y de la Dignidad`
- `Artículo 12. Libertad de Conciencia y de Religión`
- `Artículo 13. Libertad de Pensamiento y de Expresión`
- `Artículo 14. Derecho de Rectificación o Respuesta`
- `Artículo 15. Derecho de Reunión`
- `Artículo 16. Libertad de Asociación`
- `Artículo 17. Protección a la Familia`
- `Artículo 18. Derecho al Nombre`
- `Artículo 19. Derechos del Niño`
- `Artículo 20. Derecho a la Nacionalidad`
- `Artículo 22. Derecho de Circulación y de Residencia`
- `Artículo 23. Derechos Políticos`
- `Artículo 24. Igualdad ante la Ley`
- `Artículo 25. Protección Judicial`
- `Artículo 26. Desarrollo Progresivo`
- `Artículo 27. Suspensión de Garantías`
- `Artículo 28. Cláusula Federal`, `Artículo 21. Derecho a la Propiedad Privada`
- `Artículo_29_Normas_de_Interpretación`
- `Artículo 30. Alcance de las Restricciones`
- `Artículo 63.1 Reparaciones`

This model was originally trained with 6089 legal texts (see the original work [here](https://huggingface.co/hackathon-pln-es/jurisbert-clas-art-convencion-americana-dh) about the American Convention of Human Rights. It has been finetuned with the International Convention of Human Rights and other similar documents (as, for example, https://www.ohchr.org/sites/default/files/UDHR/Documents/UDHR_Translations/spn.pdf).

## Predicted Entities

`Artículo 1. Obligación de Respetar los Derechos`, `Artículo 2. Deber de Adoptar Disposiciones de Derecho Interno`, `Artículo 3. Derecho al Reconocimiento de la Personalidad Jurídica`, `Artículo 4. Derecho a la Vida`, `Artículo 5. Derecho a la Integridad Personal`, `Artículo 6. Prohibición de la Esclavitud y Servidumbre`, `Artículo 7. Derecho a la Libertad Personal`, `Artículo 8. Garantías Judiciales`, `Artículo 9. Principio de Legalidad y de Retroactividad`, `Artículo 11. Protección de la Honra y de la Dignidad`, `Artículo 12. Libertad de Conciencia y de Religión`, `Artículo 13. Libertad de Pensamiento y de Expresión`, `Artículo 14. Derecho de Rectificación o Respuesta`, `Artículo 15. Derecho de Reunión`, `Artículo 16. Libertad de Asociación`, `Artículo 17. Protección a la Familia`, `Artículo 18. Derecho al Nombre`, `Artículo 19. Derechos del Niño`, `Artículo 20. Derecho a la Nacionalidad`, `Artículo 22. Derecho de Circulación y de Residencia`, `Artículo 23. Derechos Políticos`, `Artículo 24. Igualdad ante la Ley`, `Artículo 25. Protección Judicial`, `Artículo 26. Desarrollo Progresivo`, `Artículo 27. Suspensión de Garantías`, `Artículo 28. Cláusula Federal`, `Artículo 21. Derecho a la Propiedad Privada`, `Artículo 29. Normas de Interpretación`, `Artículo 30. Alcance de las Restricciones`, `Artículo 63.1 Reparaciones`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_human_rights_en_1.0.0_3.2_1660057114857.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_human_rights_en_1.0.0_3.2_1660057114857.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler() \
       .setInputCol("text") \
       .setOutputCol("document")

sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = nlp.Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

tokenClassifier = nlp.RoBertaForSequenceClassification.pretrained("legclf_human_rights","en", "legal/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

text = """Todos los seres humanos nacen libres e iguales en dignidad y derechos y, dotados como están de razón y conciencia, deben comportarse fraternalmente los unos con los otros."""

data = spark.createDataFrame([[text]]).toDF("text")

result = pipeline.fit(data).transform(data)
```

</div>

## Results

```bash
+--------------------+-------------------------------------------------+
|                text|                                           result|
+--------------------+-------------------------------------------------+
|Todos los seres h...|[Artículo 1. Obligación de Respetar los Derechos]|
+--------------------+-------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_human_rights|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|466.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

This model was originally trained on 6089 legal texts (see the original work [here](https://huggingface.co/hackathon-pln-es/jurisbert-clas-art-convencion-americana-dh) about the American Convention of Human Rights. It has been finetuned with the International Convention of Human Rights and other similar documents (as, for example, https://www.ohchr.org/sites/default/files/UDHR/Documents/UDHR_Translations/spn.pdf).

## Benchmarking

```bash
label             precision  recall    f1-score    support
accuracy            -          -       0.91        98
macro-avg         0.92       0.91      0.91        98
weighted-avg      0.92       0.90      0.91        98
```       
