---
layout: model
title: Clinical Deidentification Pipeline (Portuguese)
author: John Snow Labs
name: clinical_deidentification
date: 2022-06-21
tags: [deid, deidentification, pt, licensed]
task: [De-identification, Pipeline Healthcare]
language: pt
edition: Healthcare NLP 3.5.0
spark_version: 3.0
supported: true
recommended: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pipeline is trained with `w2v_cc_300d` portuguese embeddings and can be used to deidentify PHI information from medical texts in Spanish. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask, fake or obfuscate the following entities: `AGE`, `DATE`, `PROFESSION`, `EMAIL`, `ID`, `COUNTRY`, `STREET`, `DOCTOR`, `HOSPITAL`, `PATIENT`, `URL`, `IP`, `ORGANIZATION`, `PHONE`, `ZIP`, `ACCOUNT`, `SSN`, `PLATE`, `SEX` and `IPADDR`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT_MULTI/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/DEID_PHI_TEXT_MULTI.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_pt_3.5.0_3.0_1655820388743.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_pt_3.5.0_3.0_1655820388743.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from johnsnowlabs import *

deid_pipeline = PretrainedPipeline("clinical_deidentification", "pt", "clinical/models")

sample = """Dados do paciente.
Nome: Mauro.
Apelido: Gonçalves.
NIF: 368503.
NISS: 26 63514095.
Endereço: Calle Miguel Benitez 90.
CÓDIGO POSTAL: 28016.
Dados de cuidados.
Data de nascimento: 03/03/1946.
País: Portugal.
Idade: 70 anos Sexo: M.
Data de admissão: 12/12/2016.
Doutor: Ignacio Navarro Cuéllar NºCol: 28 28 70973.
Relatório clínico do paciente: Paciente de 70 anos, mineiro reformado, sem alergias medicamentosas conhecidas, que apresenta como história pessoal: acidente de trabalho antigo com fracturas vertebrais e das costelas; operado por doença de Dupuytren na mão direita e iliofemoral esquerda; Diabetes Mellitus tipo II, hipercolesterolemia e hiperuricemia; alcoolismo activo, fumador de 20 cigarros / dia.
Foi encaminhado dos cuidados primários porque apresentou uma vez hematúria macroscópica pós-morte e depois microhaematúria persistente, com micturição normal.
O exame físico mostrou um bom estado geral, com abdómen e genitália normais; o exame rectal foi compatível com adenoma de próstata de grau I/IV.
A urinálise mostrou 4 glóbulos vermelhos/campo e 0-5 leucócitos/campo; o resto do sedimento estava normal.
Hemograma normal; a bioquímica mostrou glicemia de 169 mg/dl e triglicéridos de 456 mg/dl; função hepática e renal normal. PSA de 1,16 ng/ml.
A citologia da urina era repetidamente desconfiada por malignidade.
A radiografia simples abdominal mostra alterações degenerativas na coluna lombar e calcificações vasculares tanto no hipocôndrio como na pélvis.
A ecografia urológica revelou cistos corticais simples no rim direito, uma bexiga inalterada com boa capacidade e uma próstata com 30g de peso.
O IVUS mostrou normofuncionalismo renal bilateral, calcificações na silhueta renal direita e ureteres artrosados com imagens de adição no terço superior de ambos os ureteres, relacionadas com pseudodiverticulose ureteral. O cistograma mostra uma bexiga com boa capacidade, mas com paredes trabeculadas em relação à bexiga de stress. A tomografia computorizada abdominal é normal.
A cistoscopia revelou a existência de pequenos tumores na bexiga, e a ressecção transuretral foi realizada com o resultado anatomopatológico do carcinoma urotelial superficial da bexiga.
Referido por: Miguel Santos - Avenida dos Aliados, 22 Portugal E-mail: nnavcu@hotmail.com.
"""

result = deid_pipeline .annotate(sample)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val deid_pipeline = new PretrainedPipeline("clinical_deidentification", "pt", "clinical/models")

sample = "Dados do paciente.
Nome: Mauro.
Apelido: Gonçalves.
NIF: 368503.
NISS: 26 63514095.
Endereço: Calle Miguel Benitez 90.
CÓDIGO POSTAL: 28016.
Dados de cuidados.
Data de nascimento: 03/03/1946.
País: Portugal.
Idade: 70 anos Sexo: M.
Data de admissão: 12/12/2016.
Doutor: Ignacio Navarro Cuéllar NºCol: 28 28 70973.
Relatório clínico do paciente: Paciente de 70 anos, mineiro reformado, sem alergias medicamentosas conhecidas, que apresenta como história pessoal: acidente de trabalho antigo com fracturas vertebrais e das costelas; operado por doença de Dupuytren na mão direita e iliofemoral esquerda; Diabetes Mellitus tipo II, hipercolesterolemia e hiperuricemia; alcoolismo activo, fumador de 20 cigarros / dia.
Foi encaminhado dos cuidados primários porque apresentou uma vez hematúria macroscópica pós-morte e depois microhaematúria persistente, com micturição normal.
O exame físico mostrou um bom estado geral, com abdómen e genitália normais; o exame rectal foi compatível com adenoma de próstata de grau I/IV.
A urinálise mostrou 4 glóbulos vermelhos/campo e 0-5 leucócitos/campo; o resto do sedimento estava normal.
Hemograma normal; a bioquímica mostrou glicemia de 169 mg/dl e triglicéridos de 456 mg/dl; função hepática e renal normal. PSA de 1,16 ng/ml.
A citologia da urina era repetidamente desconfiada por malignidade.
A radiografia simples abdominal mostra alterações degenerativas na coluna lombar e calcificações vasculares tanto no hipocôndrio como na pélvis.
A ecografia urológica revelou cistos corticais simples no rim direito, uma bexiga inalterada com boa capacidade e uma próstata com 30g de peso.
O IVUS mostrou normofuncionalismo renal bilateral, calcificações na silhueta renal direita e ureteres artrosados com imagens de adição no terço superior de ambos os ureteres, relacionadas com pseudodiverticulose ureteral. O cistograma mostra uma bexiga com boa capacidade, mas com paredes trabeculadas em relação à bexiga de stress. A tomografia computorizada abdominal é normal.
A cistoscopia revelou a existência de pequenos tumores na bexiga, e a ressecção transuretral foi realizada com o resultado anatomopatológico do carcinoma urotelial superficial da bexiga.
Referido por: Miguel Santos - Avenida dos Aliados, 22 Portugal E-mail: nnavcu@hotmail.com"

val result = deid_pipeline.annotate(sample)
```
</div>

## Results

```bash
Masked with entity labels
------------------------------
Dados do <SEX>.
Nome: <PATIENT>.
Apelido: <PATIENT>.
NIF: <ID>.
NISS: <ID>.
Endereço: <STREET>.
CÓDIGO POSTAL: <ZIP>.
Dados de cuidados.
Data de nascimento: <DATE>.
País: <COUNTRY>.
Idade: <AGE> anos Sexo: <SEX>.
Data de admissão: <DATE>.
Doutor: <DOCTOR> Cuéllar NºCol: <ID> <ID> <ID>.
Relatório clínico do <SEX>: <SEX> de <AGE> anos, mineiro reformado, sem alergias medicamentosas conhecidas, que apresenta como história pessoal: acidente de trabalho antigo com fracturas vertebrais e das costelas; operado por doença de Dupuytren na mão direita e iliofemoral esquerda;
Diabetes Mellitus tipo II, hipercolesterolemia e hiperuricemia; alcoolismo activo, fumador de 20 cigarros / dia.
Foi encaminhado dos cuidados primários porque apresentou uma vez hematúria macroscópica pós-morte e depois microhaematúria persistente, com micturição normal.
O exame físico mostrou um bom estado geral, com abdómen e genitália normais; o exame rectal foi compatível com adenoma de próstata de grau I/IV.
A urinálise mostrou 4 glóbulos vermelhos/campo e 0-5 leucócitos/campo; o resto do sedimento estava normal.
Hemograma normal; a bioquímica mostrou glicemia de 169 mg/dl e triglicér<SEX> de 456 mg/dl; função hepática e renal normal. PSA de 1,16 ng/ml.
A citologia da urina era repetidamente desconfiada por malignidade.
A radiografia simples abdominal mostra alterações degenerativas na coluna lombar e calcificações vasculares tanto no hipocôndrio como na pélvis.
A ecografia urológica revelou cistos corticais simples no rim direito, uma bexiga inalterada com boa capacidade e uma próstata com 30g de peso.
O IVUS mostrou normofuncionalismo renal bilateral, calcificações na silhueta renal direita e ureteres artrosados com imagens de adição no terço superior de ambos os ureteres, relacionadas com pseudodiverticulose ureteral. O cistograma mostra uma bexiga com boa capacidade, mas com paredes trabeculadas em relação à bexiga de stress.
A tomografia computorizada abdominal é normal.
A cistoscopia revelou a existência de pequenos tumores na bexiga, e a ressecção transuretral foi realizada com o resultado anatomopatológico do carcinoma urotelial superficial da bexiga.
Referido por: <DOCTOR> - <STREET>, 22 <COUNTRY> E-mail: <EMAIL>.

Masked with chars
------------------------------
Dados do [******].
Nome: [***].
Apelido: [*******].
NIF: [****].
NISS: [*********].
Endereço: [*********************].
CÓDIGO POSTAL: [***].
Dados de cuidados.
Data de nascimento: [********].
País: [******].
Idade: ** anos Sexo: *.
Data de admissão: [********].
Doutor: [*************] Cuéllar NºCol: ** ** [***].
Relatório clínico do [******]: [******] de ** anos, mineiro reformado, sem alergias medicamentosas conhecidas, que apresenta como história pessoal: acidente de trabalho antigo com fracturas vertebrais e das costelas; operado por doença de Dupuytren na mão direita e iliofemoral esquerda;
Diabetes Mellitus tipo II, hipercolesterolemia e hiperuricemia; alcoolismo activo, fumador de 20 cigarros / dia.
Foi encaminhado dos cuidados primários porque apresentou uma vez hematúria macroscópica pós-morte e depois microhaematúria persistente, com micturição normal.
O exame físico mostrou um bom estado geral, com abdómen e genitália normais; o exame rectal foi compatível com adenoma de próstata de grau I/IV.
A urinálise mostrou 4 glóbulos vermelhos/campo e 0-5 leucócitos/campo; o resto do sedimento estava normal.
Hemograma normal; a bioquímica mostrou glicemia de 169 mg/dl e triglicér[**] de 456 mg/dl; função hepática e renal normal. PSA de 1,16 ng/ml.
A citologia da urina era repetidamente desconfiada por malignidade.
A radiografia simples abdominal mostra alterações degenerativas na coluna lombar e calcificações vasculares tanto no hipocôndrio como na pélvis.
A ecografia urológica revelou cistos corticais simples no rim direito, uma bexiga inalterada com boa capacidade e uma próstata com 30g de peso.
O IVUS mostrou normofuncionalismo renal bilateral, calcificações na silhueta renal direita e ureteres artrosados com imagens de adição no terço superior de ambos os ureteres, relacionadas com pseudodiverticulose ureteral. O cistograma mostra uma bexiga com boa capacidade, mas com paredes trabeculadas em relação à bexiga de stress.
A tomografia computorizada abdominal é normal.
A cistoscopia revelou a existência de pequenos tumores na bexiga, e a ressecção transuretral foi realizada com o resultado anatomopatológico do carcinoma urotelial superficial da bexiga.
Referido por: [***********] - [*****************], 22 [******] E-mail: [****************].

Masked with fixed length chars
------------------------------
Dados do ****.
Nome: ****.
Apelido: ****.
NIF: ****.
NISS: ****.
Endereço: ****.
CÓDIGO POSTAL: ****.
Dados de cuidados.
Data de nascimento: ****.
País: ****.
Idade: **** anos Sexo: ****.
Data de admissão: ****.
Doutor: **** Cuéllar NºCol: **** **** ****.
Relatório clínico do ****: **** de **** anos, mineiro reformado, sem alergias medicamentosas conhecidas, que apresenta como história pessoal: acidente de trabalho antigo com fracturas vertebrais e das costelas; operado por doença de Dupuytren na mão direita e iliofemoral esquerda;
Diabetes Mellitus tipo II, hipercolesterolemia e hiperuricemia; alcoolismo activo, fumador de 20 cigarros / dia.
Foi encaminhado dos cuidados primários porque apresentou uma vez hematúria macroscópica pós-morte e depois microhaematúria persistente, com micturição normal.
O exame físico mostrou um bom estado geral, com abdómen e genitália normais; o exame rectal foi compatível com adenoma de próstata de grau I/IV.
A urinálise mostrou 4 glóbulos vermelhos/campo e 0-5 leucócitos/campo; o resto do sedimento estava normal.
Hemograma normal; a bioquímica mostrou glicemia de 169 mg/dl e triglicér**** de 456 mg/dl; função hepática e renal normal. PSA de 1,16 ng/ml.
A citologia da urina era repetidamente desconfiada por malignidade.
A radiografia simples abdominal mostra alterações degenerativas na coluna lombar e calcificações vasculares tanto no hipocôndrio como na pélvis.
A ecografia urológica revelou cistos corticais simples no rim direito, uma bexiga inalterada com boa capacidade e uma próstata com 30g de peso.
O IVUS mostrou normofuncionalismo renal bilateral, calcificações na silhueta renal direita e ureteres artrosados com imagens de adição no terço superior de ambos os ureteres, relacionadas com pseudodiverticulose ureteral. O cistograma mostra uma bexiga com boa capacidade, mas com paredes trabeculadas em relação à bexiga de stress.
A tomografia computorizada abdominal é normal.
A cistoscopia revelou a existência de pequenos tumores na bexiga, e a ressecção transuretral foi realizada com o resultado anatomopatológico do carcinoma urotelial superficial da bexiga.
Referido por: **** - ****, 22 **** E-mail: ****.

Obfuscated
------------------------------
Dados do H..
Nome: Marcos Alves.
Apelido: Tiago Santos.
NIF: 566-445.
NISS: 134544332.
Endereço: Rua de Santa María, 100.
CÓDIGO POSTAL: 4099.
Dados de cuidados.
Data de nascimento: 31/03/1946.
País: Espanha.
Idade: 46 anos Sexo: Mulher.
Data de admissão: 06/01/2017.
Doutor: Carlos Melo Cuéllar NºCol: 134544332 134544332 124 445 311.
Relatório clínico do H.: M. de 46 anos, mineiro reformado, sem alergias medicamentosas conhecidas, que apresenta como história pessoal: acidente de trabalho antigo com fracturas vertebrais e das costelas; operado por doença de Dupuytren na mão direita e iliofemoral esquerda;
Diabetes Mellitus tipo II, hipercolesterolemia e hiperuricemia; alcoolismo activo, fumador de 20 cigarros / dia.
Foi encaminhado dos cuidados primários porque apresentou uma vez hematúria macroscópica pós-morte e depois microhaematúria persistente, com micturição normal.
O exame físico mostrou um bom estado geral, com abdómen e genitália normais; o exame rectal foi compatível com adenoma de próstata de grau I/IV.
A urinálise mostrou 4 glóbulos vermelhos/campo e 0-5 leucócitos/campo; o resto do sedimento estava normal.
Hemograma normal; a bioquímica mostrou glicemia de 169 mg/dl e triglicérHomen de 456 mg/dl; função hepática e renal normal. PSA de 1,16 ng/ml.
A citologia da urina era repetidamente desconfiada por malignidade.
A radiografia simples abdominal mostra alterações degenerativas na coluna lombar e calcificações vasculares tanto no hipocôndrio como na pélvis.
A ecografia urológica revelou cistos corticais simples no rim direito, uma bexiga inalterada com boa capacidade e uma próstata com 30g de peso.
O IVUS mostrou normofuncionalismo renal bilateral, calcificações na silhueta renal direita e ureteres artrosados com imagens de adição no terço superior de ambos os ureteres, relacionadas com pseudodiverticulose ureteral. O cistograma mostra uma bexiga com boa capacidade, mas com paredes trabeculadas em relação à bexiga de stress.
A tomografia computorizada abdominal é normal.
A cistoscopia revelou a existência de pequenos tumores na bexiga, e a ressecção transuretral foi realizada com o resultado anatomopatológico do carcinoma urotelial superficial da bexiga.
Referido por: Carlos Melo - Avenida Dos Aliados, 56, 22 Espanha E-mail: maria.prado@jacob.com.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clinical_deidentification|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.5.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|pt|
|Size:|1.3 GB|

## Included Models

- nlp.DocumentAssembler
- nlp.SentenceDetectorDLModel
- nlp.TokenizerModel
- nlp.WordEmbeddingsModel
- medical.NerModel
- nlp.NerConverter
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- nlp.TextMatcherModel
- ContextualParserModel
- ContextualParserModel
- nlp.RegexMatcherModel
- nlp.RegexMatcherModel
- ChunkMergeModel
- medical.DeIdentificationModel
- medical.DeIdentificationModel
- medical.DeIdentificationModel
- medical.DeIdentificationModel
- Finisher
