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

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.ml.param.{DoubleParam, Param, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.collection.immutable.Map

/** LLM-based Named Entity Recognition using few-shot prompting.
  *
  * LLMNerModel performs entity extraction from text using Large Language Models (LLMs) with
  * few-shot examples. The annotator expects JSON-formatted responses from an LLM (e.g., from
  * [[com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFModel AutoGGUFModel]]) and parses them to
  * extract named entities.
  *
  * This annotator follows the LangExtract pattern from Google Research, where the LLM extracts
  * entity text (without character positions), and post-processing can optionally add character
  * indices later through fuzzy matching.
  *
  * The expected LLM response format is:
  * {{{
  * {
  *   "extractions": [
  *     {"entity": "MEDICATION", "text": "aspirin", "confidence": 0.95},
  *     {"entity": "DOSAGE", "text": "250mg", "confidence": 0.92}
  *   ]
  * }
  * }}}
  *
  * Alternative formats are also supported:
  * {{{
  * {
  *   "extractions": [
  *     {"extraction_class": "MEDICATION", "extraction_text": "amoxicillin"}
  *   ]
  * }
  * }}}
  *
  * ==Note==
  * This version does not compute character indices (begin/end positions). All entities are
  * returned with `begin = -1` and `end = -1`. Character indexing will be added in a future
  * version using fuzzy string matching (similar to Google's LangExtract library).
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val llmNer = LLMNerModel.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("entities")
  * }}}
  *
  * For available pretrained models please see the [[https://sparknlp.org/models Models Hub]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFModel
  * import com.johnsnowlabs.nlp.annotators.ner.dl.LLMNerModel
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * // LLM with few-shot prompt for medical NER
  * val llm = AutoGGUFModel.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("llm_response")
  *   .setNPredict(500)
  *
  * // Parse LLM response and extract entities
  * val llmNer = new LLMNerModel()
  *   .setInputCols("llm_response")
  *   .setOutputCol("entities")
  *   .setEntityTypes(Array("MEDICATION", "DOSAGE"))
  *   .setMinConfidence(0.7)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   llm,
  *   llmNer
  * ))
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
  * @see
  *   [[https://github.com/google/langextract LangExtract]] for the pattern this follows
  * @see
  *   [[https://sparknlp.org/docs/en/annotators Annotators Main Page]] for a list of NER
  *   annotators
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
    with HasSimpleAnnotate[LLMNerModel]
    with HasProtectedParams {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("LLM_NER"))

  /** Input Annotator Type: DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)

  /** Output Annotator Type: NAMED_ENTITY
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.NAMED_ENTITY

  /** Labels mapping entity names to IDs (optional, for compatibility)
    *
    * @group param
    */
  val labels: MapFeature[String, Int] = new MapFeature(this, "labels").setProtected()

  /** @group setParam */
  def setLabels(value: Map[String, Int]): this.type = set(labels, value)

  /** @group getParam */
  def getLabels: Map[String, Int] = $$(labels)

  /** Minimum confidence threshold for including entities (Default: `0.0`)
    *
    * Entities with confidence scores below this threshold will be filtered out.
    *
    * @group param
    */
  val minConfidence =
    new DoubleParam(this, "minConfidence", "Minimum confidence threshold for including entities")

  /** @group setParam */
  def setMinConfidence(value: Double): this.type = set(minConfidence, value)

  /** @group getParam */
  def getMinConfidence: Double = $(minConfidence)

  /** List of entity types to extract (optional filter) (Default: empty = extract all)
    *
    * If set, only entities with types in this list will be returned.
    *
    * @group param
    */
  val entityTypes =
    new StringArrayParam(this, "entityTypes", "List of entity types to extract (optional filter)")

  /** @group setParam */
  def setEntityTypes(value: Array[String]): this.type = set(entityTypes, value)

  /** @group getParam */
  def getEntityTypes: Array[String] = $(entityTypes)

  /** JSON response format (Default: `"extractions"`)
    *
    * Specifies the key name in the JSON response that contains the entity array. Common values:
    *   - `"extractions"` (default, LangExtract format)
    *   - `"entities"`
    *   - `"results"`
    *
    * @group param
    */
  val jsonArrayKey =
    new Param[String](this, "jsonArrayKey", "JSON response format key for entity array")

  /** @group setParam */
  def setJsonArrayKey(value: String): this.type = set(jsonArrayKey, value)

  /** @group getParam */
  def getJsonArrayKey: String = $(jsonArrayKey)

  setDefault(
    minConfidence -> 0.0,
    entityTypes -> Array.empty[String],
    jsonArrayKey -> "extractions")

  /** Main annotation method - processes documents containing LLM JSON responses
    *
    * @param annotations
    *   Annotations (LLM responses) to process
    * @return
    *   Extracted entity annotations
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.flatMap { annotation =>
      val llmResponse = annotation.result
      // Convert metadata to immutable Map
      val immutableMetadata: Map[String, String] = annotation.metadata.toMap
      // Parse entities from the LLM JSON response
      parseEntitiesFromResponse(llmResponse, immutableMetadata)
    }
  }

  /** Parse entity annotations from LLM JSON response
    *
    * @param llmResponse
    *   Raw LLM output text
    * @param sourceMetadata
    *   Metadata from the source annotation
    * @return
    *   Sequence of entity annotations
    */
  private def parseEntitiesFromResponse(
      llmResponse: String,
      sourceMetadata: Map[String, String]): Seq[Annotation] = {
    try {
      // Step 1: Extract JSON (handles fenced code blocks like ```json ... ```)
      val jsonText = extractJsonFromResponse(llmResponse)

      // Step 2: Parse JSON
      implicit val formats: DefaultFormats.type = DefaultFormats
      val parsed = parse(jsonText)

      // Step 3: Extract the entity array using the configured key
      val extractions = (parsed \ $(jsonArrayKey)).extract[List[Map[String, Any]]]

      // Step 4: Convert to Annotations
      val annotations = extractions.zipWithIndex.flatMap { case (extraction, idx) =>
        createAnnotationFromExtraction(extraction, idx, sourceMetadata)
      }

      // Step 5: Filter by entity types if specified
      val filteredByType = if ($(entityTypes).nonEmpty) {
        annotations.filter { ann =>
          $(entityTypes).contains(ann.metadata.getOrElse("entity", ""))
        }
      } else {
        annotations
      }

      // Step 6: Filter by confidence threshold
      if ($(minConfidence) > 0.0) {
        filteredByType.filter { ann =>
          ann.metadata.get("confidence").exists(_.toDouble >= $(minConfidence))
        }
      } else {
        filteredByType
      }

    } catch {
      case e: Exception =>
        // Log the error but don't fail the entire pipeline
        System.err.println(s"[LLMNerModel] Failed to parse LLM response: ${e.getMessage}")
        Seq.empty[Annotation]
    }
  }

  /** Create an Annotation from a single extraction
    *
    * @param extraction
    *   Map containing extraction data
    * @param index
    *   Position in the extractions array
    * @param sourceMetadata
    *   Metadata from source annotation
    * @return
    *   Optional Annotation
    */
  private def createAnnotationFromExtraction(
      extraction: Map[String, Any],
      index: Int,
      sourceMetadata: Map[String, String]): Option[Annotation] = {

    // Support both "entity" and "extraction_class" keys
    val entityType = extraction
      .get("entity")
      .orElse(extraction.get("extraction_class"))
      .map(_.toString)

    // Support both "text" and "extraction_text" keys
    val entityText = extraction
      .get("text")
      .orElse(extraction.get("extraction_text"))
      .map(_.toString)

    // Both entity type and text are required
    (entityType, entityText) match {
      case (Some(entity), Some(text)) if text.nonEmpty =>
        // Get optional attributes (e.g., "route", "frequency" for medical entities)
        val attributes = extraction.get("attributes") match {
          case Some(attrs: Map[_, _] @unchecked) =>
            attrs.map { case (k, v) => k.toString -> v.toString }
          case _ => Map.empty[String, String]
        }

        // Get optional confidence (default to 1.0)
        val confidence = extraction.get("confidence") match {
          case Some(conf: Double) => conf.toString
          case Some(conf: Int) => conf.toDouble.toString
          case Some(conf: String) => conf
          case _ => "1.0"
        }

        // Create metadata
        val metadata = Map(
          "entity" -> entity,
          "confidence" -> confidence,
          "index" -> index.toString) ++ attributes ++ sourceMetadata

        // Create annotation without character positions (begin/end = -1)
        // This will be updated in a future version with fuzzy string matching
        Some(
          Annotation(
            annotatorType = AnnotatorType.NAMED_ENTITY,
            begin = -1, // No character indexing yet
            end = -1, // Will be added later with fuzzy matching
            result = text,
            metadata = metadata))

      case _ =>
        System.err.println(
          s"[LLMNerModel] Extraction missing required fields (entity/text): $extraction")
        None
    }
  }

  /** Extract JSON from LLM response (handles fenced code blocks)
    *
    * Many LLMs wrap their JSON output in markdown code fences like:
    * {{{
    * ```json
    * { "extractions": [...] }
    * ```
    * }}}
    *
    * This method handles both fenced and raw JSON.
    *
    * @param response
    *   Raw LLM response
    * @return
    *   Cleaned JSON string
    */
  private def extractJsonFromResponse(response: String): String = {
    // Pattern for fenced JSON blocks: ```json ... ```
    val fencePattern = """```json\s*(.*?)\s*```""".r

    fencePattern.findFirstMatchIn(response) match {
      case Some(m) => m.group(1).trim
      case None =>
        // No fence found, try to find raw JSON
        val startIdx = response.indexOf("{")
        val endIdx = response.lastIndexOf("}")

        if (startIdx != -1 && endIdx != -1 && endIdx > startIdx) {
          response.substring(startIdx, endIdx + 1)
        } else {
          // Return as-is and let JSON parser handle the error
          response.trim
        }
    }
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    // No model weights to save - this is just a JSON parser
  }
}

trait ReadablePretrainedLLMNerModel
    extends ParamsAndFeaturesReadable[LLMNerModel]
    with HasPretrained[LLMNerModel] {
  override val defaultModelName: Some[String] = Some("llm_ner_base")
  override val defaultLang: String = "en"

  /** Java compliant-overrides */
  override def pretrained(): LLMNerModel = super.pretrained()

  override def pretrained(name: String): LLMNerModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): LLMNerModel =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): LLMNerModel =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadLLMNerModel { this: ParamsAndFeaturesReadable[LLMNerModel] =>

  def readModel(instance: LLMNerModel, path: String, spark: SparkSession): Unit = {
    // No model weights to read - this is just a JSON parser
    // Configuration is handled by parent class
  }

  addReader(readModel)
}

/** This is the companion object of [[LLMNerModel]]. Please refer to that class for the
  * documentation.
  */
object LLMNerModel extends ReadablePretrainedLLMNerModel with ReadLLMNerModel
