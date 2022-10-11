{%- capture title -%}
NerConverterInternal
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Converts a IOB or IOB2 representation of NER to a user-friendly one,
by associating the tokens of recognized entities and their label.
Chunks with no associated entity (tagged "O") are filtered.
See also [Inside–outside–beginning (tagging)](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) for more information.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, TOKEN, NAMED_ENTITY
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_python_medical -%}
from johnsnowlabs import * 

documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") 

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

jsl_ner = medical.NerModel.pretrained("ner_jsl", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("jsl_ner")

jsl_ner_converter = nlp.NerConverter() \
    .setInputCols(["sentence", "token", "jsl_ner"]) \
    .setOutputCol("jsl_ner_chunk")

jsl_ner_converter_internal = medical.NerConverterInternal()\
    .setInputCols(["sentence","token","jsl_ner"])\
    .setOutputCol("replaced_ner_chunk")\
    .setReplaceDictResource("replace_dict.csv","text", {"delimiter":","})
      
nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    jsl_ner,
    jsl_ner_converter,
    jsl_ner_converter_internal
    ])

result = nlpPipeline.fit(data).transform(data)

{%- endcapture -%}

{%- capture model_scala_medical -%}
from johnsnowlabs import * 
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = nlp.SentenceDetectorDLModel
    .pretrained("sentence_detector_dl_healthcare","en","clinical/models") 
    .setInputCols("document") 
    .setOutputCol("sentence") 


val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

 
val word_embeddings = nlp.WordEmbeddingsModel
   .pretrained("embeddings_clinical", "en","clinical/models")
   .setInputCols(Array("sentence", "token"))
   .setOutputCol("embeddings")


val jsl_ner = medical.NerModel
    .pretrained("ner_jsl", "en", "clinical/models") 
    .setInputCols(Array("sentence", "token","embeddings")) 
    .setOutputCol("jsl_ner")


val jsl_ner_converter = new nlp.NerConverter() 
    .setInputCols(Array("sentence", "token", "jsl_ner")) 
    .setOutputCol("jsl_ner_chunk")

val jsl_ner_converter_internal = new medical.NerConverterInternal() 
    .setInputCols(Array("sentence", "token", "jsl_ner")) 
    .setOutputCol("replaced_ner_chunk")
    .setReplaceDictResource("replace_dict.csv","text", {"delimiter":","})


val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  word_embeddings,
  jsl_ner,
  jsl_ner_converter,
  jsl_ner_converter_internal

))

val result = pipeline.fit(data).transform(data)
{%- endcapture -%}


{%- capture model_python_legal -%}
from johnsnowlabs import * 

documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")
    #.setCustomBounds(["\n\n"])

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

legal_ner = legal.NerModel.pretrained("legner_contract_doc_parties", "en", "legal/models")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner") 
    #.setLabelCasing("upper")

ner_converter = legal.NerConverterInternal() \
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\
    .setReplaceLabels({"ALIAS": "PARTY"}) # "ALIAS" are secondary names of companies, so let's extract them also as PARTY

nlpPipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      embeddings,
      legal_ner,
      ner_converter])

result = nlpPipeline.fit(data).transform(data)
{%- endcapture -%}



{%- capture model_scala_legal -%}
from johnsnowlabs import * 
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = nlp.SentenceDetectorDLModel
    .pretrained("sentence_detector_dl","xx") 
    .setInputCols("document") 
    .setOutputCol("sentence") 


val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

 
val embeddings = nlp.RoBertaEmbeddings
   .pretrained("roberta_embeddings_legal_roberta_base", "en")
   .setInputCols(Array("sentence", "token"))
   .setOutputCol("embeddings")


val legal_ner = legal.NerModel
    .pretrained("legner_contract_doc_parties", "en", "legal/models") 
    .setInputCols(Array("sentence", "token","embeddings")) 
    .setOutputCol("ner")


val ner_converter = new legal.NerConverterInternal() 
    .setInputCols(Array("sentence", "token", "ner")) 
    .setOutputCol("ner_chunk")
    .setReplaceLabels({"ALIAS": "PARTY"})


val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  legal_ner,
  ner_converter
))

val result = pipeline.fit(data).transform(data)

{%- endcapture -%}




{%- capture model_python_finance -%}
from johnsnowlabs import * 

documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector =  nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")
    #.setCustomBounds(["\n\n"])

tokenizer =  nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings =  nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

fin_ner = finance.NerModel.pretrained("finner_deid", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner") 
    #.setLabelCasing("upper")

ner_converter = finance.NerConverterInternal() \
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\
    .setReplaceLabels({"ORG": "PARTY"}) # Replace "ORG" entity as "PARTY"

nlpPipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      embeddings,
      fin_ner,
      ner_converter])

result = nlpPipeline.fit(data).transform(data)


{%- endcapture -%}


{%- capture model_scala_finance -%}
from johnsnowlabs import * 
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = nlp.SentenceDetectorDLModel
    .pretrained("sentence_detector_dl","xx") 
    .setInputCols("document")
    .setOutputCol("sentence") 


val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

 
val embeddings = nlp.RoBertaEmbeddings
   .pretrained("roberta_embeddings_legal_roberta_base", "en")
   .setInputCols(Array("sentence", "token"))
   .setOutputCol("embeddings")


val fin_ner = finance.NerModel
    .pretrained("finner_deid", "en", "finance/models") 
    .setInputCols(Array("sentence", "token","embeddings")) 
    .setOutputCol("ner")


val ner_converter = new finance.NerConverterInternal() 
    .setInputCols(Array("sentence", "token", "ner")) 
    .setOutputCol("ner_chunk")
    .setReplaceLabels({"ORG": "PARTY"}) 


val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  fin_ner,
  ner_converter
))

val result = pipeline.fit(data).transform(data)


{%- endcapture -%}




{%- capture model_api_link -%}
[NerConverterInternal](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/NerConverterInternal)
{%- endcapture -%}

{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_scala_medical=model_scala_medical
model_python_legal=model_python_legal
model_scala_legal=model_scala_legal
model_python_finance=model_python_finance
model_scala_finance=model_scala_finance
model_api_link=model_api_link%}