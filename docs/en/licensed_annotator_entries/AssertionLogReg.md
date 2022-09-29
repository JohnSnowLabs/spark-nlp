{%- capture title -%}
AssertionLogReg
{%- endcapture -%}

{%- capture approach -%}
approach
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
This is a main class in AssertionLogReg family. Logarithmic Regression is used to extract Assertion Status
from extracted entities and text. AssertionLogRegModel requires DOCUMENT, CHUNK and WORD_EMBEDDINGS type
annotator inputs, which can be obtained by e.g a
[DocumentAssembler](/docs/en/annotators#documentassembler),
[NerConverter](/docs/en/annotators#nerconverter)
and [WordEmbeddingsModel](/docs/en/annotators#wordembeddings).
The result is an assertion status annotation for each recognized entity.
Possible values are `"Negated", "Affirmed" and "Historical"`.

Unlike the DL Model, this class does not extend AnnotatorModel.
Instead it extends the RawAnnotator, that's why the main point of interest is method transform().

At the moment there are no pretrained models available for this class. Please refer to AssertionLogRegApproach to
train your own model.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, CHUNK, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture model_output_anno -%}
ASSERTION
{%- endcapture -%}

{%- capture model_api_link -%}
[AssertionLogRegModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/logreg/AssertionLogRegModel)
{%- endcapture -%}

{%- capture approach_description -%}
Trains a classification method, which uses the Logarithmic Regression Algorithm. It is used to extract Assertion Status
from extracted entities and text.
Contains all the methods for training a AssertionLogRegModel, together with trainWithChunk, trainWithStartEnd.
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT, CHUNK, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_output_anno -%}
ASSERTION
{%- endcapture -%}

{%- capture approach_python_medical -%}
from johnsnowlabs import *
# Training with Glove Embeddings
# First define pipeline stages to extract embeddings and text chunks
documentAssembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = nlp.Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

glove = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("word_embeddings") \
    .setCaseSensitive(False)

chunkAssembler = nlp.Doc2Chunk() \
    .setInputCols(["document"]) \
    .setChunkCol("target") \
    .setOutputCol("chunk")

# Then the AssertionLogRegApproach model is defined. Label column is needed in the dataset for training.
assertion = medical.AssertionLogRegApproach() \
    .setLabelCol("label") \
    .setInputCols(["document", "chunk", "word_embeddings"]) \
    .setOutputCol("assertion") \
    .setReg(0.01) \
    .setBefore(11) \
    .setAfter(13) \
    .setStartCol("start") \
    .setEndCol("end")

assertionPipeline = Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    embeddings,
    nerModel,
    nerConverter,
    assertion
])

assertionModel = assertionPipeline.fit(dataset)
{%- endcapture -%}

{%- capture approach_scala_medical -%}
from johnsnowlabs import * 

// Training with Glove Embeddings
// First define pipeline stages to extract embeddings and text chunks
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new nlp.Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val glove = nlp.WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("document", "token"))
  .setOutputCol("word_embeddings")
  .setCaseSensitive(false)

val chunkAssembler = new nlp.Doc2Chunk()
  .setInputCols("document")
  .setChunkCol("target")
  .setOutputCol("chunk")

// Then the AssertionLogRegApproach model is defined. Label column is needed in the dataset for training.
val assertion = new medical.AssertionLogRegApproach()
  .setLabelCol("label")
  .setInputCols(Array("document", "chunk", "word_embeddings"))
  .setOutputCol("assertion")
  .setReg(0.01)
  .setBefore(11)
  .setAfter(13)
  .setStartCol("start")
  .setEndCol("end")

val assertionPipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  nerModel,
  nerConverter,
  assertion
))

val assertionModel = assertionPipeline.fit(dataset)
{%- endcapture -%}

{%- capture approach_api_link -%}
[AssertionLogRegApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/logreg/AssertionLogRegApproach)
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
approach_scala_medical=approach_scala_medical
approach_api_link=approach_api_link
%}
