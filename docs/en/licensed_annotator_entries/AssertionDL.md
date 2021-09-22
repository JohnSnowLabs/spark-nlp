{%- capture title -%}
AssertionDL
{%- endcapture -%}

{%- capture model_description -%}
AssertionDL is a deep Learning based approach used to extract Assertion Status
from extracted entities and text. AssertionDLModel requires DOCUMENT, CHUNK and WORD_EMBEDDINGS type
annotator inputs, which can be obtained by e.g a
[DocumentAssembler](/docs/en/annotators#documentassembler),
[NerConverter](/docs/en/annotators#nerconverter)
and [WordEmbeddingsModel](/docs/en/annotators#wordembeddings).
The result is an assertion status annotation for each recognized entity.
Possible values include `“present”, “absent”, “hypothetical”, “conditional”, “associated_with_other_person”` etc.

For pretrained models please see the
[Models Hub](https://nlp.johnsnowlabs.com/models?task=Assertion+Status) for available models.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, CHUNK, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture model_output_anno -%}
ASSERTION
{%- endcapture -%}

{%- capture model_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
# Define pipeline stages to extract NER chunks first
data = spark.createDataFrame([
  ["Patient with severe fever and sore throat"],
  ["Patient shows no stomach pain"],
  ["She was maintained on an epidural and PCA for pain control."]]).toDF("text")
documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
  .setOutputCol("embeddings")
nerModel = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]).setOutputCol("ner")
nerConverter = NerConverter().setInputCols(["sentence", "token", "ner"]).setOutputCol("ner_chunk")

# Then a pretrained AssertionDLModel is used to extract the assertion status
clinicalAssertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
  .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
  .setOutputCol("assertion")

assertionPipeline = Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  nerModel,
  nerConverter,
  clinicalAssertion
])

assertionModel = assertionPipeline.fit(data)

# Show results
result = assertionModel.transform(data)
result.selectExpr("ner_chunk.result", "assertion.result").show(3, truncate=False)
+--------------------------------+--------------------------------+
|result                          |result                          |
+--------------------------------+--------------------------------+
|[severe fever, sore throat]     |[present, present]              |
|[stomach pain]                  |[absent]                        |
|[an epidural, PCA, pain control]|[present, present, hypothetical]|
+--------------------------------+--------------------------------+

{%- endcapture -%}

{%- capture model_scala_example -%}
// Define pipeline stages to extract NER chunks first
val data = Seq(
  "Patient with severe fever and sore throat",
  "Patient shows no stomach pain",
  "She was maintained on an epidural and PCA for pain control.").toDF("text")
val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token")
val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setOutputCol("embeddings")
val nerModel = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
  .setInputCols("sentence", "token", "embeddings").setOutputCol("ner")
val nerConverter = new NerConverter().setInputCols("sentence", "token", "ner").setOutputCol("ner_chunk")

// Then a pretrained AssertionDLModel is used to extract the assertion status
val clinicalAssertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models")
  .setInputCols("sentence", "ner_chunk", "embeddings")
  .setOutputCol("assertion")

val assertionPipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  nerModel,
  nerConverter,
  clinicalAssertion
))

val assertionModel = assertionPipeline.fit(data)

// Show results
val result = assertionModel.transform(data)
result.selectExpr("ner_chunk.result", "assertion.result").show(3, truncate=false)
+--------------------------------+--------------------------------+
|result                          |result                          |
+--------------------------------+--------------------------------+
|[severe fever, sore throat]     |[present, present]              |
|[stomach pain]                  |[absent]                        |
|[an epidural, PCA, pain control]|[present, present, hypothetical]|
+--------------------------------+--------------------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[AssertionDLModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/dl/AssertionDLModel)
{%- endcapture -%}

{%- capture approach_description -%}
Trains AssertionDL, a deep Learning based approach used to extract Assertion Status
from extracted entities and text.
Contains all the methods for training an AssertionDLModel.
For pretrained models please use AssertionDLModel and see the
[Models Hub](https://nlp.johnsnowlabs.com/models?task=Assertion+Status) for available models.
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT, CHUNK, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_output_anno -%}
ASSERTION
{%- endcapture -%}

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
# First, pipeline stages for pre-processing the dataset (containing columns for text and label) are defined.
document = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
chunk = Doc2Chunk() \
    .setInputCols(["document"]) \
    .setOutputCol("chunk")
token = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")
embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")

# Define AssertionDLApproach with parameters and start training
assertionStatus = AssertionDLApproach() \
    .setLabelCol("label") \
    .setInputCols(["document", "chunk", "embeddings"]) \
    .setOutputCol("assertion") \
    .setBatchSize(128) \
    .setDropout(0.012) \
    .setLearningRate(0.015) \
    .setEpochs(1) \
    .setStartCol("start") \
    .setEndCol("end") \
    .setMaxSentLen(250)

trainingPipeline = Pipeline().setStages([
    document,
    chunk,
    token,
    embeddings,
    assertionStatus
])

assertionModel = trainingPipeline.fit(data)
assertionResults = assertionModel.transform(data).cache()

{%- endcapture -%}

{%- capture approach_scala_example -%}
// First, pipeline stages for pre-processing the dataset (containing columns for text and label) are defined.
val document = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")
val chunk = new Doc2Chunk()
  .setInputCols("document")
  .setOutputCol("chunk")
val token = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")
val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols("document", "token")
  .setOutputCol("embeddings")

// Define AssertionDLApproach with parameters and start training
val assertionStatus = new AssertionDLApproach()
  .setLabelCol("label")
  .setInputCols("document", "chunk", "embeddings")
  .setOutputCol("assertion")
  .setBatchSize(128)
  .setDropout(0.012f)
  .setLearningRate(0.015f)
  .setEpochs(1)
  .setStartCol("start")
  .setEndCol("end")
  .setMaxSentLen(250)

val trainingPipeline = new Pipeline().setStages(Array(
  document,
  chunk,
  token,
  embeddings,
  assertionStatus
))

val assertionModel = trainingPipeline.fit(data)
val assertionResults = assertionModel.transform(data).cache()

{%- endcapture -%}

{%- capture approach_api_link -%}
[AssertionDLApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/dl/AssertionDLApproach)
{%- endcapture -%}


{% include templates/licensed_approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_example=model_python_example
model_scala_example=model_scala_example
model_api_link=model_api_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_api_link=approach_api_link
%}
