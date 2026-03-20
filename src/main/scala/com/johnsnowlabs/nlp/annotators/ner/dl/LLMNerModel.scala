/*
 * Copyright 2017-2024 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.ml.gguf.GGUFWrapper
import com.johnsnowlabs.ml.gguf.GGUFWrapper.findGGUFModelInFolder
import com.johnsnowlabs.ml.util.LlamaCPP
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.llama.LlamaExtensions
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import de.kherud.llama.{InferenceParameters, LlamaException, LlamaModel}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, IntParam, Param, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SparkSession}
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.util.matching.Regex

/** End-to-end LLM-based Named Entity Recognition using AutoGGUF with BNF grammars.
  *
  * LLMNerModel is an end-to-end annotator that performs entity extraction from text using Large
  * Language Models (LLMs) with structured JSON output via BNF grammars. It embeds AutoGGUFModel
  * directly and uses simple string matching to compute character indices for extracted entities.
  *
  * This annotator follows the LangExtract pattern from Google Research, combining few-shot
  * prompting with constrained generation through llama.cpp BNF grammars to ensure valid JSON
  * output.
  *
  * The LLM generates responses in this format (enforced by grammar):
  * {{{
  * {
  *   "extractions": [
  *     {"entity": "MEDICATION", "text": "aspirin"},
  *     {"entity": "DOSAGE", "text": "250mg"}
  *   ]
  * }
  * }}}
  *
  * The annotator then performs string matching to find the character positions of each entity in
  * the original text, outputting CHUNK annotations with accurate begin/end indices.
  *
  * Batch processing is used for performance - all documents are processed together in a single
  * LLM call via multiComplete for maximum throughput.
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotators.ner.dl.LLMNerModel
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val llmNer = LLMNerModel
  *   .pretrained("qwen3_4b_bf16_gguf")
  *   .setInputCols("document")
  *   .setOutputCol("entities")
  *   .setEntityTypes(Array("MEDICATION", "DOSAGE", "ROUTE", "FREQUENCY"))
  *   .setNPredict(500)
  *   .setTemperature(0.1f)
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, llmNer))
  *
  * val data = Seq("Patient prescribed 500mg amoxicillin PO TID").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("entities.result", "entities.metadata").show(false)
  * +------------------------------+--------------------------------+
  * |result                        |metadata                        |
  * +------------------------------+--------------------------------+
  * |[500mg, amoxicillin, PO, TID] |[{entity -> DOSAGE}, ...]       |
  * +------------------------------+--------------------------------+
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param 1
  * @groupprio anno 2
  * @groupprio Ungrouped 3
  * @groupprio setParam 4
  * @groupprio getParam 5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class LLMNerModel(override val uid: String)
    extends AnnotatorModel[LLMNerModel]
    with HasBatchedAnnotate[LLMNerModel]
    with HasLlamaCppModelProperties
    with HasLlamaCppInferenceProperties
    with HasProtectedParams
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("LLM_NER"))

  /** Input Annotator Type: DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)

  /** Output Annotator Type: CHUNK
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.CHUNK

  private var _model: Option[Broadcast[GGUFWrapper]] = None

  val defaultGrammar =
    """root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

ws ::= ([ \t\n\r])*"""

  val defaultPrompt =
    """You are an expert entity extraction assistant. Extract entities from the following text.

Entity types to extract: {entityTypes}

IMPORTANT: Extract the EXACT text as it appears in the input. Output ONLY valid JSON.

Output format:
{
  "extractions": [
    {"entity": "<type>", "text": "<exact_text>"}
  ]
}
{examples}
Text to analyze:

Output:"""

  /** @group getParam */
  def getModelIfNotSet: GGUFWrapper = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, wrapper: GGUFWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(spark.sparkContext.broadcast(wrapper))
    }
    this
  }

  /** Closes the llama.cpp model backend freeing resources. The model is reloaded when used again.
    */
  def close(): Unit = GGUFWrapper.closeBroadcastModel(_model)

  /** Custom prompt template for NER extraction (Default: medical NER prompt)
    *
    * The prompt should include instructions for the LLM to extract entities in JSON format. Use
    * {entityTypes} as a placeholder for the entity types list.
    *
    * @group param
    */
  val promptTemplate = new Param[String](
    this,
    "promptTemplate",
    "Custom prompt template for NER extraction. Use {entityTypes} placeholder.")

  /** @group setParam */
  def setPromptTemplate(value: String): this.type = set(promptTemplate, value)

  /** @group getParam */
  def getPromptTemplate: String = $(promptTemplate)

  /** List of entity types to extract (Default: general types)
    *
    * These entity types are used in the prompt to guide the LLM's extraction.
    *
    * @group param
    */
  val entityTypes =
    new StringArrayParam(this, "entityTypes", "List of entity types to extract (used in prompt)")

  /** @group setParam */
  def setEntityTypes(value: Array[String]): this.type = set(entityTypes, value)

  /** @group getParam */
  def getEntityTypes: Array[String] = $(entityTypes)

  /** Case sensitivity for entity matching (Default: `false`)
    *
    * When false, entity matching is case-insensitive.
    *
    * @group param
    */
  val caseSensitive =
    new BooleanParam(this, "caseSensitive", "Whether entity matching is case-sensitive")

  /** @group setParam */
  def setCaseSensitive(value: Boolean): this.type = set(caseSensitive, value)

  /** @group getParam */
  def getCaseSensitive: Boolean = $(caseSensitive)

  /** Few-shot examples for the prompt (Default: empty array)
    *
    * Each example should be a tuple of (input_text, json_output). These examples will be inserted
    * into the prompt to help the LLM understand the expected output format.
    *
    * @group param
    */
  val fewShotExamples = new Param[Array[(String, String)]](
    this,
    "fewShotExamples",
    "Few-shot examples as array of (input, output_json) tuples")

  /** @group setParam */
  def setFewShotExamples(value: Array[(String, String)]): this.type = set(fewShotExamples, value)

  /** Java/Python-compatible setter for fewShotExamples.
    *
    * When called from PySpark via py4j, Python lists of tuples arrive as
    * java.util.ArrayList[java.util.ArrayList[String]]. This overload converts them to the
    * expected Scala Array[(String, String)].
    *
    * @group setParam
    */
  def setFewShotExamples(value: java.util.List[java.util.List[String]]): this.type = {
    import scala.collection.JavaConverters._
    val converted = value.asScala.flatMap { inner =>
      val list = inner.asScala
      if (list.size == 2) Some((list(0), list(1)))
      else None
    }.toArray
    set(fewShotExamples, converted)
  }

  /** @group getParam */
  def getFewShotExamples: Array[(String, String)] = $(fewShotExamples)

  private[johnsnowlabs] def setEngine(engineName: String): this.type = set(engine, engineName)

  // Default values
  setDefault(
    entityTypes -> Array("PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME"),
    caseSensitive -> false,
    promptTemplate -> defaultPrompt,
    fewShotExamples -> Array.empty[(String, String)],
    engine -> LlamaCPP.name,
    useChatTemplate -> true,
    nCtx -> 4096,
    nBatch -> 512,
    nPredict -> 500,
    nGpuLayers -> 99,
    temperature -> 0.1f,
    batchSize -> 4,
    reasoningBudget -> 0,
    grammar -> defaultGrammar)

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    getModelIfNotSet.saveToFile(path)
  }

  /** Build the system prompt with NER instructions and few-shot examples */
  private def buildSystemPrompt(): String = {
    val entityTypesStr = $(entityTypes).mkString(", ")

    val examples = getFewShotExamplesSafe
    val examplesSection = if (examples.nonEmpty) {
      val exampleTexts = examples
        .map { case (input, output) =>
          s"""Example:
Input: "$input"
Output: ```json
$output
```"""
        }
        .mkString("\n\n")
      s"\n\n$exampleTexts"
    } else {
      ""
    }

    val baseInstructions = $(promptTemplate)
      .replace("{entityTypes}", entityTypesStr)
      .replace("{examples}", examplesSection)
      .trim
    baseInstructions
  }

  /** Batch annotation method - processes all documents through AutoGGUF
    *
    * @param batchedAnnotations
    *   Batched annotations (documents) to process
    * @return
    *   Extracted entity annotations with character indices for each document
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    val annotations: Seq[Annotation] = batchedAnnotations.flatten

    if (annotations.isEmpty) {
      return batchedAnnotations.map(_ => Seq.empty[Annotation])
    }

    if (!isSet(systemPrompt) || $(systemPrompt).isEmpty) {
      set(systemPrompt, buildSystemPrompt())
    }

    val text = annotations.map(annotation => annotation.result).toArray

    val modelParams = getModelParameters.setParallel(getBatchSize)
    val inferenceParams: InferenceParameters = getInferenceParameters
    val model: LlamaModel = getModelIfNotSet.getSession(modelParams)

    val llmResponses: Array[String] =
      try {
        LlamaExtensions.multiComplete(model, inferenceParams, $(systemPrompt), text)
      } catch {
        case e: LlamaException =>
          logger.error("Error in llama.cpp batch completion", e)
          Array.fill(text.length)("""{"extractions": []}""")
      }

    val allResults: Seq[Seq[Annotation]] =
      annotations.zip(llmResponses).map { case (annotation, llmResponse) =>
        val documentText = annotation.result
        val documentBegin = annotation.begin
        val entities = parseEntitiesFromResponse(llmResponse)

        matchEntitiesToText(entities, documentText, documentBegin, annotation.metadata.toMap)
      }

    var resultIndex = 0
    batchedAnnotations.map { batch =>
      val batchResults = allResults.slice(resultIndex, resultIndex + batch.length).flatten
      resultIndex += batch.length
      batchResults
    }
  }

  /** Safely retrieve fewShotExamples, handling both native Scala Array[(String, String)] and */
  private def getFewShotExamplesSafe: Array[(String, String)] = {
    try {
      val raw = getOrDefault(fewShotExamples).asInstanceOf[Any]
      raw match {
        case arr: Array[_] =>
          arr.flatMap {
            case (a: String, b: String) => Some((a, b))
            case tuple: Product if tuple.productArity == 2 =>
              Some((tuple.productElement(0).toString, tuple.productElement(1).toString))
            case _ => None
          }
        case _ => Array.empty[(String, String)]
      }
    } catch {
      case _: Exception => Array.empty[(String, String)]
    }
  }

  /** Parse entity annotations from LLM JSON response */
  private def parseEntitiesFromResponse(llmResponse: String): Seq[EntityExtraction] = {
    try {
      // Step 1: Extract JSON (handles fenced code blocks like ```json ... ```)
      val jsonText = extractJsonFromResponse(llmResponse)

      // Step 2: Parse JSON
      implicit val formats: DefaultFormats.type = DefaultFormats
      val parsed =
        try {
          parse(jsonText)
        } catch {
          case e: Exception =>
            // JSON may be truncated (nPredict exhausted) - attempt repair
            repairTruncatedJson(jsonText) match {
              case Some(repaired) =>
                if (logger != null)
                  logger.warn("LLM JSON was truncated, recovered partial entities. " +
                    "Consider increasing nPredict or setting reasoningBudget to 0 for long texts.")
                parse(repaired)
              case None =>
                val errorMsg = if (e.getMessage != null) e.getMessage else "Unknown error"
                if (logger != null)
                  logger.error(s"Failed to parse or repair LLM JSON: $errorMsg")
                return Seq.empty[EntityExtraction]
            }
        }

      // Step 3: Extract the entity array
      val extractions =
        (parsed \ "extractions").extract[List[scala.collection.immutable.Map[String, Any]]]

      // Step 4: Convert to EntityExtraction case class
      val allExtractions = extractions.flatMap { extraction =>
        val entityType = extraction.get("entity").map(_.toString)
        val entityText = extraction.get("text").map(_.toString)

        // Extract additional attributes if present
        val additionalAttributes = extraction.get("attributes") match {
          case Some(attrs: scala.collection.immutable.Map[_, _]) =>
            attrs.map { case (k, v) => (k.toString, v.toString) }.toMap
          case _ => Map.empty[String, String]
        }

        (entityType, entityText) match {
          case (Some(entity), Some(text)) if text.nonEmpty =>
            Some(EntityExtraction(entity, text, additionalAttributes))
          case _ => None
        }
      }

      // Step 5: Filter by entity types
      val entityTypeSet = $(entityTypes).toSet
      allExtractions
        .filter(e => entityTypeSet.isEmpty || entityTypeSet.contains(e.entityType))
    } catch {
      case e: Exception =>
        val errorMsg = if (e != null && e.getMessage != null) e.getMessage else "Unknown error"
        val stackTrace = if (e != null) e.getStackTrace.take(3).mkString("; ") else ""
        if (logger != null) {
          logger.error(s"Failed to parse LLM response: $errorMsg. Stack: $stackTrace")
        } else {
          System.err.println(s"Failed to parse LLM response: $errorMsg. Stack: $stackTrace")
        }
        Seq.empty[EntityExtraction]
    }
  }

  /** Match entities to their positions in the original text with chunk indexing
    *
    * This method creates CHUNK annotations with proper begin/end indices and metadata similar to
    * other Spark NLP annotators like Chunker and NerConverter.
    */
  private def matchEntitiesToText(
      entities: Seq[EntityExtraction],
      documentText: String,
      documentBegin: Int,
      sourceMetadata: scala.collection.immutable.Map[String, String]): Seq[Annotation] = {

    val searchText = if ($(caseSensitive)) documentText else documentText.toLowerCase
    var lastSearchPosition = 0
    var chunkIndex = 0

    entities.flatMap { entity =>
      val entityTextToSearch = if ($(caseSensitive)) entity.text else entity.text.toLowerCase

      val startIdx = searchText.indexOf(entityTextToSearch, lastSearchPosition)

      if (startIdx >= 0) {
        val endIdx = startIdx + entity.text.length - 1
        val actualText = documentText.substring(startIdx, endIdx + 1)

        val metadata = scala.collection.immutable.Map(
          "entity" -> entity.entityType,
          "chunk" -> chunkIndex.toString) ++ entity.attributes ++ sourceMetadata

        lastSearchPosition = endIdx + 1
        chunkIndex += 1

        Some(
          Annotation(
            annotatorType = AnnotatorType.CHUNK,
            begin = documentBegin + startIdx,
            end = documentBegin + endIdx,
            result = actualText,
            metadata = metadata))
      } else {
        val fallbackIdx = searchText.indexOf(entityTextToSearch)
        if (fallbackIdx >= 0) {
          val endIdx = fallbackIdx + entity.text.length - 1
          val actualText = documentText.substring(fallbackIdx, endIdx + 1)

          val metadata = scala.collection.immutable.Map(
            "entity" -> entity.entityType,
            "chunk" -> chunkIndex.toString) ++ entity.attributes ++ sourceMetadata

          chunkIndex += 1

          Some(
            Annotation(
              annotatorType = AnnotatorType.CHUNK,
              begin = documentBegin + fallbackIdx,
              end = documentBegin + endIdx,
              result = actualText,
              metadata = metadata))
        } else {

          None
        }
      }
    }
  }

  /** Extract JSON from LLM response (handles fenced code blocks and escape sequences) */
  private def extractJsonFromResponse(response: String): String = {
    // Pattern for fenced JSON blocks: ```json ... ``` (using (?s) DOTALL mode for multiline)
    val fencePattern: Regex = """(?s)```json\s*(.*?)\s*```""".r

    val rawJson = fencePattern.findFirstMatchIn(response) match {
      case Some(m) => m.group(1).trim
      case None =>
        // Try simpler fence pattern: ``` ... ```
        val simpleFencePattern: Regex = """(?s)```\s*(.*?)\s*```""".r
        simpleFencePattern.findFirstMatchIn(response) match {
          case Some(m) => m.group(1).trim
          case None =>
            // No fence found, try to find raw JSON by matching balanced braces
            val startIdx = response.indexOf("{")
            val endIdx = response.lastIndexOf("}")

            if (startIdx != -1 && endIdx != -1 && endIdx > startIdx) {
              response.substring(startIdx, endIdx + 1)
            } else {
              response.trim
            }
        }
    }

    cleanJsonString(rawJson)
  }

  /** Attempt to repair truncated JSON output from the LLM.
    *
    * When nPredict is exhausted, the JSON may be cut off mid-array or mid-object. This method
    * tries to find the last complete entity object and close the JSON structure properly.
    */
  private def repairTruncatedJson(json: String): Option[String] = {
    val lastCompleteObjEnd = {
      val pattern = """\"text\"\s*:\s*\"[^"]*\"\s*\}""".r
      val matches = pattern.findAllMatchIn(json).toList
      if (matches.nonEmpty) matches.last.end else -1
    }

    if (lastCompleteObjEnd > 0) {
      val truncated = json.substring(0, lastCompleteObjEnd).trim
      val repaired = if (truncated.endsWith("},")) {
        truncated.dropRight(1) + "]}"
      } else if (truncated.endsWith("}")) {
        truncated + "]}"
      } else {
        return None
      }
      Some(repaired)
    } else {
      None
    }
  }

  /** Clean JSON string to handle various escape sequence issues from LLM output */
  private def cleanJsonString(json: String): String = {
    // Step 1: Remove ALL literal backslash-quote sequences (\") outside of proper JSON strings
    val step1 = json
      .replace("\\\"", "\"")
      .replace("\\r\\n", " ")
      .replace("\\n", " ")
      .replace("\\r", " ")
      .replace("\\t", " ")

    // Step 2: Normalize actual whitespace characters (the real bytes)
    val normalized = step1
      .replace("\r\n", " ")
      .replace("\r", " ")
      .replace("\n", " ")
      .replace("\t", " ")
      .replaceAll("\\s+", " ")
      .trim

    normalized
  }
}

/** Case class for entity extraction */
case class EntityExtraction(
    entityType: String,
    text: String,
    attributes: Map[String, String] = Map.empty)

trait ReadablePretrainedLLMNerModel
    extends ParamsAndFeaturesFallbackReadable[LLMNerModel]
    with HasPretrained[LLMNerModel] {
  override val defaultModelName: Some[String] = Some("qwen3_4b_bf16_gguf")
  override val defaultLang: String = "en"

  /** Java compliant-overrides */
  override def pretrained(): LLMNerModel = super.pretrained()

  override def pretrained(name: String): LLMNerModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): LLMNerModel =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): LLMNerModel =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadLLMNerModel {
  this: ParamsAndFeaturesFallbackReadable[LLMNerModel] =>

  /** Fallback: load a raw GGUF folder (e.g. from AutoGGUFModel.save) */
  override def fallbackLoad(folder: String, spark: SparkSession): LLMNerModel = {
    val actualFolderPath: String = ResourceHelper.resolvePath(folder)
    val localFolder = ResourceHelper.copyToLocal(actualFolderPath)
    val modelFile = findGGUFModelInFolder(localFolder)
    loadSavedModel(modelFile, spark)
  }

  /** Reader called by Spark after params are deserialized */
  def readModel(instance: LLMNerModel, path: String, spark: SparkSession): Unit = {
    val model: GGUFWrapper = GGUFWrapper.readModel(path, spark)
    instance.setModelIfNotSet(spark, model)
  }

  addReader(readModel)

  /** Load a GGUF model file from a local path into a new LLMNerModel */
  def loadSavedModel(modelPath: String, spark: SparkSession): LLMNerModel = {
    val localPath: String = ResourceHelper.copyToLocal(modelPath)
    val annotatorModel = new LLMNerModel()
    annotatorModel
      .setModelIfNotSet(spark, GGUFWrapper.read(spark, localPath))
      .setEngine(LlamaCPP.name)

    val metadata = LlamaExtensions.getMetadataFromFile(localPath)
    if (metadata.nonEmpty) annotatorModel.setMetadata(metadata)
    annotatorModel
  }
}

object LLMNerModel extends ReadablePretrainedLLMNerModel with ReadLLMNerModel
