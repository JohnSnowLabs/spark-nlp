{%- capture title -%}
NerDisambiguator
{%- endcapture -%}

{%- capture approach -%}
approach
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Links words of interest, such as names of persons, locations and companies, from an input text document to
a corresponding unique entity in a target Knowledge Base (KB). Words of interest are called Named Entities (NEs),
mentions, or surface forms.
Instantiated / pretrained model of the NerDisambiguator.
Links words of interest, such as names of persons, locations and companies, from an input text document to
a corresponding unique entity in a target Knowledge Base (KB). Words of interest are called Named Entities (NEs),
mentions, or surface forms.
{%- endcapture -%}

{%- capture model_input_anno -%}
CHUNK, SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture model_output_anno -%}
DISAMBIGUATION
{%- endcapture -%}

{%- capture model_api_link -%}
[NerDisambiguatorModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/disambiguation/NerDisambiguatorModel)
{%- endcapture -%}

{%- capture approach_description -%}
Links words of interest, such as names of persons, locations and companies, from an input text document to
a corresponding unique entity in a target Knowledge Base (KB). Words of interest are called Named Entities (NEs),
mentions, or surface forms.
The model needs extracted CHUNKS and SENTENCE_EMBEDDINGS type input from e.g.
[SentenceEmbeddings](/docs/en/annotators#sentenceembeddings) and
[NerConverter](/docs/en/annotators#nerconverter).
{%- endcapture -%}

{%- capture approach_input_anno -%}
CHUNK, SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_output_anno -%}
DISAMBIGUATION
{%- endcapture -%}

{%- capture approach_python_medical -%}
from johnsnowlabs import *
# Extracting Person identities
# First define pipeline stages that extract entities and embeddings. Entities are filtered for PER type entities.
# Extracting Person identities
# First define pipeline stages that extract entities and embeddings. Entities are filtered for PER type entities.
data = spark.createDataFrame([["The show also had a contestant named Donald Trump who later defeated Christina Aguilera ..."]]) \
  .toDF("text")
documentAssembler = nlp.DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")
sentenceDetector = nlp.SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentence")
tokenizer = nlp.Tokenizer() \
  .setInputCols(["sentence"]) \
  .setOutputCol("token")
word_embeddings = nlp.WordEmbeddingsModel.pretrained() \
  .setInputCols(["sentence", "token"]) \
  .setOutputCol("embeddings")
sentence_embeddings = nlp.SentenceEmbeddings() \
  .setInputCols(["sentence","embeddings"]) \
  .setOutputCol("sentence_embeddings")
ner_model = nlp.NerDLModel.pretrained() \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
ner_converter = nlp.NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk") \
  .setWhiteList(["PER"])

# Then the extracted entities can be disambiguated.
disambiguator = medical.NerDisambiguator() \
  .setS3KnowledgeBaseName("i-per") \
  .setInputCols(["ner_chunk", "sentence_embeddings"]) \
  .setOutputCol("disambiguation") \
  .setNumFirstChars(5)

nlpPipeline = Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  tokenizer,
  word_embeddings,
  sentence_embeddings,
  ner_model,
  ner_converter,
  disambiguator])

model = nlpPipeline.fit(data)
result = model.transform(data)

# Show results
result.selectExpr("explode(disambiguation)")
  .selectExpr("col.metadata.chunk as chunk", "col.result as result").show(5, False)
+------------------+------------------------------------------------------------------------------------------------------------------------+
|chunk             |result                                                                                                                  |
+------------------+------------------------------------------------------------------------------------------------------------------------+
|Donald Trump      |http:#en.wikipedia.org/?curid=4848272, http:#en.wikipedia.org/?curid=31698421, http:#en.wikipedia.org/?curid=55907961   |
|Christina Aguilera|http:#en.wikipedia.org/?curid=144171, http:#en.wikipedia.org/?curid=6636454                                             |
+------------------+------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture approach_python_legal -%}
from johnsnowlabs import *
# Extracting Person identities
# First define pipeline stages that extract entities and embeddings. Entities are filtered for PER type entities.
# Extracting Person identities
# First define pipeline stages that extract entities and embeddings. Entities are filtered for PER type entities.

documentAssembler = nlp.DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")
sentenceDetector = nlp.SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentence")
tokenizer = nlp.Tokenizer() \
  .setInputCols(["sentence"]) \
  .setOutputCol("token")
word_embeddings = nlp.WordEmbeddingsModel.pretrained() \
  .setInputCols(["sentence", "token"]) \
  .setOutputCol("embeddings")
sentence_embeddings = nlp.SentenceEmbeddings() \
  .setInputCols(["sentence","embeddings"]) \
  .setOutputCol("sentence_embeddings")
ner_model = nlp.NerDLModel.pretrained() \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
ner_converter = nlp.NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk") \
  .setWhiteList(["PER"])

# Then the extracted entities can be disambiguated.
disambiguator = legal.NerDisambiguator() \
  #.setS3KnowledgeBaseName("i-per") \
  .setInputCols(["ner_chunk", "sentence_embeddings"]) \
  .setOutputCol("disambiguation") \
  .setNumFirstChars(5)

nlpPipeline = Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  tokenizer,
  word_embeddings,
  sentence_embeddings,
  ner_model,
  ner_converter,
  disambiguator])

{%- endcapture -%}


{%- capture approach_python_finance -%}
from johnsnowlabs import *
# Extracting Person identities
# First define pipeline stages that extract entities and embeddings. Entities are filtered for PER type entities.
# Extracting Person identities
# First define pipeline stages that extract entities and embeddings. Entities are filtered for PER type entities.

documentAssembler = nlp.DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")
sentenceDetector = nlp.SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentence")
tokenizer = nlp.Tokenizer() \
  .setInputCols(["sentence"]) \
  .setOutputCol("token")
word_embeddings = nlp.WordEmbeddingsModel.pretrained() \
  .setInputCols(["sentence", "token"]) \
  .setOutputCol("embeddings")
sentence_embeddings = nlp.SentenceEmbeddings() \
  .setInputCols(["sentence","embeddings"]) \
  .setOutputCol("sentence_embeddings")
ner_model = nlp.NerDLModel.pretrained() \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
ner_converter = nlp.NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk") \
  .setWhiteList(["PER"])

# Then the extracted entities can be disambiguated.
disambiguator = finance.NerDisambiguator() \
  #.setS3KnowledgeBaseName("i-per") \
  .setInputCols(["ner_chunk", "sentence_embeddings"]) \
  .setOutputCol("disambiguation") \
  .setNumFirstChars(5)

nlpPipeline = Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  tokenizer,
  word_embeddings,
  sentence_embeddings,
  ner_model,
  ner_converter,
  disambiguator])

{%- endcapture -%}



{%- capture approach_scala_medical -%}
from johnsnowlabs import * 
// Extracting Person identities
// First define pipeline stages that extract entities and embeddings. Entities are filtered for PER type entities.
val data = Seq("The show also had a contestant named Donald Trump who later defeated Christina Aguilera ...")
  .toDF("text")
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")
val sentenceDetector = new nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")
val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")
val word_embeddings = nlp.WordEmbeddingsModel.pretrained()
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val sentence_embeddings = new nlp.SentenceEmbeddings()
  .setInputCols(Array("sentence","embeddings"))
  .setOutputCol("sentence_embeddings")
val ner_model = nlp.NerDLModel.pretrained()
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")
val ner_converter = new nlp.NerConverter()
  .setInputCols(Array("sentence", "token", "ner"))
  .setOutputCol("ner_chunk")
  .setWhiteList("PER")

// Then the extracted entities can be disambiguated.
val disambiguator = new medical.NerDisambiguator()
  .setS3KnowledgeBaseName("i-per")
  .setInputCols(Array("ner_chunk", "sentence_embeddings"))
  .setOutputCol("disambiguation")
  .setNumFirstChars(5)

val nlpPipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  word_embeddings,
  sentence_embeddings,
  ner_model,
  ner_converter,
  disambiguator))

val model = nlpPipeline.fit(data)
val result = model.transform(data)

// Show results
//
// result.selectExpr("explode(disambiguation)")
//   .selectExpr("col.metadata.chunk as chunk", "col.result as result").show(5, false)
// +------------------+------------------------------------------------------------------------------------------------------------------------+
// |chunk             |result                                                                                                                  |
// +------------------+------------------------------------------------------------------------------------------------------------------------+
// |Donald Trump      |http://en.wikipedia.org/?curid=4848272, http://en.wikipedia.org/?curid=31698421, http://en.wikipedia.org/?curid=55907961|
// |Christina Aguilera|http://en.wikipedia.org/?curid=144171, http://en.wikipedia.org/?curid=6636454                                           |
// +------------------+------------------------------------------------------------------------------------------------------------------------+
//
{%- endcapture -%}


{%- capture approach_scala_legal -%}
from johnsnowlabs import * 
// Extracting Person identities
// First define pipeline stages that extract entities and embeddings. Entities are filtered for PER type entities.
val data = Seq("The show also had a contestant named Donald Trump who later defeated Christina Aguilera ...")
  .toDF("text")
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")
val sentenceDetector = new nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")
val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")
val word_embeddings = nlp.WordEmbeddingsModel.pretrained()
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val sentence_embeddings = new nlp.SentenceEmbeddings()
  .setInputCols(Array("sentence","embeddings"))
  .setOutputCol("sentence_embeddings")
val ner_model = nlp.NerDLModel.pretrained()
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")
val ner_converter = new nlp.NerConverter()
  .setInputCols(Array("sentence", "token", "ner"))
  .setOutputCol("ner_chunk")
  .setWhiteList("PER")

// Then the extracted entities can be disambiguated.
val disambiguator = new legal.NerDisambiguator()
  #.setS3KnowledgeBaseName("i-per")
  .setInputCols(Array("ner_chunk", "sentence_embeddings"))
  .setOutputCol("disambiguation")
  .setNumFirstChars(5)

val nlpPipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  word_embeddings,
  sentence_embeddings,
  ner_model,
  ner_converter,
  disambiguator))

{%- endcapture -%}

{%- capture approach_scala_finance -%}
from johnsnowlabs import * 
// Extracting Person identities
// First define pipeline stages that extract entities and embeddings. Entities are filtered for PER type entities.
val data = Seq("The show also had a contestant named Donald Trump who later defeated Christina Aguilera ...")
  .toDF("text")
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")
val sentenceDetector = new nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")
val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")
val word_embeddings = nlp.WordEmbeddingsModel.pretrained()
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val sentence_embeddings = new nlp.SentenceEmbeddings()
  .setInputCols(Array("sentence","embeddings"))
  .setOutputCol("sentence_embeddings")
val ner_model = nlp.NerDLModel.pretrained()
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")
val ner_converter = new nlp.NerConverter()
  .setInputCols(Array("sentence", "token", "ner"))
  .setOutputCol("ner_chunk")
  .setWhiteList("PER")

// Then the extracted entities can be disambiguated.
val disambiguator = new finance.NerDisambiguator()
  #.setS3KnowledgeBaseName("i-per")
  .setInputCols(Array("ner_chunk", "sentence_embeddings"))
  .setOutputCol("disambiguation")
  .setNumFirstChars(5)

val nlpPipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  word_embeddings,
  sentence_embeddings,
  ner_model,
  ner_converter,
  disambiguator))

{%- endcapture -%}




{%- capture approach_api_link -%}
[NerDisambiguator](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/disambiguation/NerDisambiguator)
{%- endcapture -%}



{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
approach=approach
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_api_link=model_api_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_medical=approach_python_medical
approach_python_legal=approach_python_legal
approach_python_finance=approach_python_finance
approach_scala_medical=approach_scala_medical
approach_scala_legal=approach_scala_legal
approach_scala_finance=approach_scala_finance
approach_api_link=approach_api_link
%}

