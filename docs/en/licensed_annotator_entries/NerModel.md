{%- capture title -%}
NerModel
{%- endcapture -%}

{%- capture approach -%}
approach
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
This Named Entity recognition annotator is a generic NER model based on Neural Networks.


Pretrained models can be loaded with `pretrained` of the companion object:
```
val nerModel = nlp.NerDLModel.pretrained()
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")
```
The default model is `"ner_clinical"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition).
Additionally, pretrained pipelines are available for this module, see [Pipelines](https://nlp.johnsnowlabs.com/docs/en/pipelines).

Note that some pretrained models require specific types of embeddings, depending on which they were trained on.
For example, the default model `"ner_dl"` requires the
[WordEmbeddings](/docs/en/annotators#wordembeddings) `"ner_clinical"`.


For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb)
(sections starting with `Training a Clinical NER`)

{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, TOKEN, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture model_output_anno -%}
NAMED_ENTITY
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

jsl_ner_pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    jsl_ner,
    jsl_ner_converter])

result = jsl_ner_pipeline.fit(data).transform(data)


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
   .pretrained("embeddings_clinical", "en", "clinical/models")
   .setInputCols(Array("sentence", "token"))
   .setOutputCol("embeddings")

val jsl_ner = medical.NerModel
    .pretrained("ner_jsl", "en", "clinical/models") 
    .setInputCols(Array("sentence", "token","embeddings")) 
    .setOutputCol("jsl_ner")

val jsl_ner_converter = new nlp.NerConverter() 
    .setInputCols(Array("sentence", "token", "jsl_ner")) 
    .setOutputCol("jsl_ner_chunk")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  word_embeddings,
  jsl_ner,
  jsl_ner_converter
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

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained("legner_headers", "en", "legal/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
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


val ner_model = legal.NerModel
    .pretrained("legner_headers", "en", "legal/models") 
    .setInputCols(Array("sentence", "token","embeddings")) 
    .setOutputCol("ner")


val ner_converter = new nlp.NerConverter() 
    .setInputCols(Array("sentence", "token", "ner")) 
    .setOutputCol("ner_chunk")




val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  ner_model,
  ner_converter
))

val result = pipeline.fit(data).transform(data)


{%- endcapture -%}





{%- capture model_python_finance -%}
from johnsnowlabs import * 

documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained("finner_headers", "en", "finance/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
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

 
val embeddings = nlp.BertEmbeddings
   .pretrained("bert_embeddings_sec_bert_base", "en")
   .setInputCols(Array("sentence", "token"))
   .setOutputCol("embeddings")


val ner_model = finance.NerModel
    .pretrained("finner_headers", "en", "finance/models") 
    .setInputCols(Array("sentence", "token","embeddings")) 
    .setOutputCol("ner")


val ner_converter = new nlp.NerConverter() 
    .setInputCols(Array("sentence", "token", "ner")) 
    .setOutputCol("ner_chunk")


val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  embeddings,
  ner_model,
  ner_converter
))

val result = pipeline.fit(data).transform(data)


{%- endcapture -%}

{%- capture model_api_link -%}
[MedicalNerModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/MedicalNerModel.html)
{%- endcapture -%}







{%- capture approach_description -%}
This Named Entity recognition annotator allows to train generic NER model based on Neural Networks.

The architecture of the neural network is a Char CNNs - BiLSTM - CRF that achieves state-of-the-art in most datasets.

For instantiated/pretrained models, see NerDLModel.

The training data should be a labeled Spark Dataset, in the format of [CoNLL](/docs/en/training#conll-dataset)
2003 IOB with `Annotation` type columns. The data should have columns of type `DOCUMENT, TOKEN, WORD_EMBEDDINGS` and an
additional label column of annotator type `NAMED_ENTITY`.
Excluding the label, this can be done with for example
- a [SentenceDetector](/docs/en/annotators#sentencedetector),
- a [Tokenizer](/docs/en/annotators#tokenizer) and
- a [WordEmbeddingsModel](/docs/en/annotators#wordembeddings) with clinical embeddings
  (any [clinical word embeddings](https://nlp.johnsnowlabs.com/models?task=Embeddings) can be chosen).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb)
(sections starting with `Training a Clinical NER`)

{%- endcapture -%}



{%- capture approach_input_anno -%}
DOCUMENT, TOKEN, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture approach_python_medical -%}
from johnsnowlabs import * 

# First extract the prerequisites for the NerDLApproach
documentAssembler = nlp.DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentence = nlp.SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

tokenizer = nlp.Tokenizer() \
.setInputCols(["sentence"]) \
.setOutputCol("token")

clinical_embeddings = nlp.WordEmbeddingsModel.pretrained('embeddings_clinical', "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

# Then the training can start
nerTagger = medical.NerApproach()\
.setInputCols(["sentence", "token", "embeddings"])\
.setLabelColumn("label")\
.setOutputCol("ner")\
.setMaxEpochs(2)\
.setBatchSize(64)\
.setRandomSeed(0)\
.setVerbose(1)\
.setValidationSplit(0.2)\
.setEvaluationLogExtended(True) \
.setEnableOutputLogs(True)\
.setIncludeConfidence(True)\
.setOutputLogsPath('ner_logs')\
.setGraphFolder('medical_ner_graphs')\
.setEnableMemoryOptimizer(True) #>> if you have a limited memory and a large conll file, you can set this True to train batch by batch

pipeline = Pipeline().setStages([
documentAssembler,
sentence,
tokenizer,
clinical_embeddings,
nerTagger
])

# We use the text and labels from the CoNLL dataset
conll = CoNLL()
trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")

pipelineModel = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture approach_python_legal -%}
from johnsnowlabs import * 

# First extract the prerequisites for the NerDLApproach
documentAssembler = nlp.DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentence = nlp.SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

tokenizer = nlp.Tokenizer() \
.setInputCols(["sentence"]) \
.setOutputCol("token")

clinical_embeddings = nlp.WordEmbeddingsModel.pretrained('embeddings_clinical', "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

# Then the training can start
nerTagger = legal.NerApproach()\
.setInputCols(["sentence", "token", "embeddings"])\
.setLabelColumn("label")\
.setOutputCol("ner")\
.setMaxEpochs(2)\
.setBatchSize(64)\
.setRandomSeed(0)\
.setVerbose(1)\
.setValidationSplit(0.2)\
.setEvaluationLogExtended(True) \
.setEnableOutputLogs(True)\
.setIncludeConfidence(True)\
.setOutputLogsPath('ner_logs')\
.setGraphFolder('medical_ner_graphs')\
.setEnableMemoryOptimizer(True) #>> if you have a limited memory and a large conll file, you can set this True to train batch by batch

pipeline = Pipeline().setStages([
documentAssembler,
sentence,
tokenizer,
clinical_embeddings,
nerTagger
])
{%- endcapture -%}

{%- capture approach_python_finance -%}
from johnsnowlabs import * 

# First extract the prerequisites for the NerDLApproach
documentAssembler = nlp.DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentence = nlp.SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

tokenizer = nlp.Tokenizer() \
.setInputCols(["sentence"]) \
.setOutputCol("token")

clinical_embeddings = nlp.WordEmbeddingsModel.pretrained('embeddings_clinical', "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

# Then the training can start
nerTagger = finance.NerApproach()\
.setInputCols(["sentence", "token", "embeddings"])\
.setLabelColumn("label")\
.setOutputCol("ner")\
.setMaxEpochs(2)\
.setBatchSize(64)\
.setRandomSeed(0)\
.setVerbose(1)\
.setValidationSplit(0.2)\
.setEvaluationLogExtended(True) \
.setEnableOutputLogs(True)\
.setIncludeConfidence(True)\
.setOutputLogsPath('ner_logs')\
.setGraphFolder('medical_ner_graphs')\
.setEnableMemoryOptimizer(True) #>> if you have a limited memory and a large conll file, you can set this True to train batch by batch

pipeline = Pipeline().setStages([
documentAssembler,
sentence,
tokenizer,
clinical_embeddings,
nerTagger
])
{%- endcapture -%}



{%- capture approach_scala_medical -%}
from johnsnowlabs import * 
// First extract the prerequisites for the NerDLApproach
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = nlp.WordEmbeddingsModel
  .pretrained('embeddings_clinical', "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")

// Then the training can start
val nerTagger =new medical.NerApproach()
.setInputCols(Array("sentence", "token", "embeddings"))
.setLabelColumn("label")
.setOutputCol("ner")
.setMaxEpochs(5)
.setLr(0.003f)
.setBatchSize(8)
.setRandomSeed(0)
.setVerbose(1)
.setEvaluationLogExtended(false)
.setEnableOutputLogs(false)
.setIncludeConfidence(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  embeddings,
  nerTagger
))

// We use the text and labels from the CoNLL dataset
val conll = CoNLL()
val trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")

val pipelineModel = pipeline.fit(trainingData)

{%- endcapture -%}


{%- capture approach_scala_legal -%}
from johnsnowlabs import * 
// First extract the prerequisites for the NerDLApproach
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = nlp.WordEmbeddingsModel
  .pretrained('embeddings_clinical', "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")

// Then the training can start
val nerTagger =new legal.NerApproach()
.setInputCols(Array("sentence", "token", "embeddings"))
.setLabelColumn("label")
.setOutputCol("ner")
.setMaxEpochs(5)
.setLr(0.003f)
.setBatchSize(8)
.setRandomSeed(0)
.setVerbose(1)
.setEvaluationLogExtended(false)
.setEnableOutputLogs(false)
.setIncludeConfidence(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  embeddings,
  nerTagger
))
{%- endcapture -%}

{%- capture approach_scala_finance -%}
from johnsnowlabs import * 
// First extract the prerequisites for the NerDLApproach
val documentAssembler = new nlp.DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new nlp.SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new nlp.Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val embeddings = nlp.WordEmbeddingsModel
  .pretrained('embeddings_clinical', "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")

// Then the training can start
val nerTagger =new finance.NerApproach()
.setInputCols(Array("sentence", "token", "embeddings"))
.setLabelColumn("label")
.setOutputCol("ner")
.setMaxEpochs(5)
.setLr(0.003f)
.setBatchSize(8)
.setRandomSeed(0)
.setVerbose(1)
.setEvaluationLogExtended(false)
.setEnableOutputLogs(false)
.setIncludeConfidence(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  embeddings,
  nerTagger
))
{%- endcapture -%}


{%- capture approach_api_link -%}
[MedicalNerApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/ner/MedicalNerApproach.html)
{%- endcapture -%}



{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
approach=approach
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_scala_medical=model_scala_medical
model_python_legal=model_python_legal
model_scala_legal=model_scala_legal
model_python_finance=model_python_finance
model_scala_finance=model_scala_finance
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


