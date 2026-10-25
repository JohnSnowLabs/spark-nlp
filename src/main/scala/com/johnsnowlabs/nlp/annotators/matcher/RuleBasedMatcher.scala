/*
 * Copyright 2017-2026 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.matcher

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.rulematcher.{
  AlignmentMode,
  OverlapStrategy,
  RuleMatcherEngine,
  RulePatternParser
}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, AnnotatorType}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{Param, Params, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

private[matcher] trait RuleBasedMatcherParams extends Params {
  this: com.johnsnowlabs.nlp.HasInputAnnotationCols =>

  private val builtInAttributes: Set[String] = Set(
    "TEXT",
    "TOKEN",
    "LOWER",
    "LENGTH",
    "POS",
    "LEMMA",
    "NER",
    "NER_TAG",
    "NER_TYPE",
    "ENT_TYPE",
    "DEP",
    "HEAD",
    "HEAD_BEGIN",
    "HEAD_END",
    "DEP_LABEL",
    "RELATION")

  private val expectedTypesByAttribute: Map[String, Set[String]] = Map(
    "TEXT" -> Set(TOKEN),
    "TOKEN" -> Set(TOKEN),
    "LOWER" -> Set(TOKEN),
    "LENGTH" -> Set(TOKEN),
    "LEMMA" -> Set(TOKEN),
    "POS" -> Set(POS),
    "NER" -> Set(NAMED_ENTITY),
    "NER_TAG" -> Set(NAMED_ENTITY),
    "NER_TYPE" -> Set(NAMED_ENTITY),
    "ENT_TYPE" -> Set(NAMED_ENTITY),
    "DEP" -> Set(DEPENDENCY),
    "HEAD" -> Set(DEPENDENCY),
    "HEAD_BEGIN" -> Set(DEPENDENCY),
    "HEAD_END" -> Set(DEPENDENCY),
    "DEP_LABEL" -> Set(LABELED_DEPENDENCY),
    "RELATION" -> Set(LABELED_DEPENDENCY))

  val attributeColumns: StringArrayParam = new StringArrayParam(
    this,
    "attributeColumns",
    "Attribute to input column mappings, encoded as ATTRIBUTE=column")

  val alignmentMode: Param[String] = new Param[String](
    this,
    "alignmentMode",
    s"Annotation alignment mode. Supported: ${AlignmentMode.values.mkString(", ")}")

  val overlapStrategy: Param[String] = new Param[String](
    this,
    "overlapStrategy",
    s"Overlap strategy. Supported: ${OverlapStrategy.values.mkString(", ")}")

  setDefault(
    attributeColumns -> Array.empty[String],
    alignmentMode -> AlignmentMode.Strict,
    overlapStrategy -> OverlapStrategy.All)

  override def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  def setAttributeColumns(value: Map[String, String]): this.type =
    set(attributeColumns, value.map { case (attr, col) => s"$attr=$col" }.toArray)

  def setAttributeColumns(value: Array[String]): this.type = set(attributeColumns, value)

  def getAttributeColumns: Map[String, String] =
    parseKeyValueEntries($(attributeColumns), "attributeColumns")

  def setAlignmentMode(value: String): this.type = {
    val normalized = value.toUpperCase
    require(
      AlignmentMode.values.contains(normalized),
      s"alignmentMode must be one of ${AlignmentMode.values.mkString(", ")}")
    set(alignmentMode, normalized)
  }

  def getAlignmentMode: String = $(alignmentMode)

  def setOverlapStrategy(value: String): this.type = {
    val normalized = value.toUpperCase
    require(
      OverlapStrategy.values.contains(normalized),
      s"overlapStrategy must be one of ${OverlapStrategy.values.mkString(", ")}")
    set(overlapStrategy, normalized)
  }

  def getOverlapStrategy: String = $(overlapStrategy)

  protected def parseKeyValueEntries(
      entries: Array[String],
      paramName: String,
      normalizeKeys: Boolean = true): Map[String, String] = {
    val pairs = entries
      .filter(_.nonEmpty)
      .map { entry =>
        val parts = entry.split("=", 2)
        require(
          parts.length == 2 && parts(0).nonEmpty && parts(1).nonEmpty,
          s"$paramName entries must use key=value format, got '$entry'")
        val key = parts(0).trim
        val parsedKey = if (normalizeKeys) key.toUpperCase else key
        parsedKey -> parts(1).trim
      }
    val duplicates = pairs.groupBy(_._1).filter(_._2.length > 1).keys.toSeq.sorted
    require(
      duplicates.isEmpty,
      s"$paramName contains duplicate attribute mappings: ${duplicates.mkString(", ")}")
    pairs.toMap
  }

  protected def validateInputColumns(schema: StructType): Boolean = {
    validateInputColumnsOrThrow(schema)
    true
  }

  protected def validateInputColumnsOrThrow(schema: StructType): Unit = {
    val configuredInputCols = get(inputCols).getOrElse(Array.empty[String])
    require(
      configuredInputCols.nonEmpty,
      "RuleBasedMatcher requires inputCols. Configure at least one DOCUMENT column " +
        "for sentence/document context and one TOKEN column for base tokens.")

    val missingCols = configuredInputCols.filterNot(schema.fieldNames.contains)
    require(
      missingCols.isEmpty,
      s"RuleBasedMatcher inputCols reference missing dataset columns: ${missingCols.mkString(", ")}. " +
        s"Available columns: ${schema.fieldNames.mkString(", ")}")

    val fieldsByName = schema.fields.map(field => field.name -> field).toMap
    val nonAnnotationCols =
      configuredInputCols.filterNot(col => fieldsByName(col).metadata.contains("annotatorType"))
    require(
      nonAnnotationCols.isEmpty,
      s"RuleBasedMatcher inputCols must be Spark NLP annotation columns with annotatorType metadata. " +
        s"Columns without annotatorType metadata: ${nonAnnotationCols.mkString(", ")}")

    val wrongSchemaCols =
      configuredInputCols.filterNot(col => fieldsByName(col).dataType == Annotation.arrayType)
    require(
      wrongSchemaCols.isEmpty,
      s"RuleBasedMatcher inputCols must use the standard Spark NLP Annotation array schema. " +
        s"Unsupported annotation schemas: ${wrongSchemaCols.mkString(", ")}")

    val imageCols = configuredInputCols.filter(col =>
      fieldsByName(col).metadata.getString("annotatorType") == IMAGE)
    require(
      imageCols.isEmpty,
      s"RuleBasedMatcher does not support IMAGE annotation columns. Remove: ${imageCols.mkString(", ")}")

    val types = columnTypes(schema)
    require(
      types.values.exists(_ == DOCUMENT),
      "RuleBasedMatcher requires a DOCUMENT input column for sentence/document boundaries. " +
        "Use a DocumentAssembler or SentenceDetector output column in inputCols.")
    require(
      types.values.exists(_ == TOKEN),
      "RuleBasedMatcher requires a TOKEN input column for base token alignment. " +
        "Use a Tokenizer output column in inputCols and map TEXT/TOKEN to it when needed.")

    val tokenCols = types.filter(_._2 == TOKEN).keys.toSeq
    val mappings = getAttributeColumns
    require(
      tokenCols.length <= 1 || Seq("TEXT", "TOKEN", "LOWER").exists(mappings.contains),
      s"RuleBasedMatcher found multiple TOKEN input columns (${tokenCols.mkString(", ")}). " +
        "Map TEXT, TOKEN, or LOWER to the base token column with setAttributeColumns.")

    val missingMappedCols = mappings.values.filterNot(configuredInputCols.contains).toSeq.distinct
    require(
      missingMappedCols.isEmpty,
      s"RuleBasedMatcher attributeColumns reference columns that are not in inputCols: " +
        s"${missingMappedCols.mkString(", ")}. Add them to inputCols or fix the mapping.")

    mappings.foreach { case (attribute, col) =>
      expectedTypesByAttribute.get(attribute).foreach { expected =>
        val actual = types(col)
        require(
          expected.contains(actual),
          s"RuleBasedMatcher attribute '$attribute' is mapped to column '$col' with annotatorType '$actual', " +
            s"but expected one of ${expected.mkString(", ")}. Fix setAttributeColumns or use a compatible annotator output.")
      }
    }
  }

  protected def columnTypes(schema: StructType): Map[String, String] =
    getInputCols.map { col =>
      val field = schema.fields.find(_.name == col).getOrElse {
        throw new IllegalArgumentException(s"Input column '$col' does not exist")
      }
      col -> field.metadata.getString("annotatorType")
    }.toMap

  protected def validateRuleAttributesOrThrow(
      rules: Seq[com.johnsnowlabs.nlp.annotators.common.rulematcher.MatcherRule],
      schema: StructType): Unit = {
    val attrs = RulePatternParser.ruleAttributes(rules).map(_.trim.toUpperCase)
    val mapped = getAttributeColumns.keySet
    val available = builtInAttributes ++ mapped
    val unknown = attrs.filterNot(attr => available.contains(attr) || attr.startsWith("META."))
    require(
      unknown.isEmpty,
      s"RuleBasedMatcher rules reference unknown attributes: ${unknown.toSeq.sorted.mkString(", ")}. " +
        s"Supported built-in attributes: ${builtInAttributes.toSeq.sorted.mkString(", ")}. " +
        "For custom annotation metadata use META.<key>, or map a custom attribute with setAttributeColumns.")

    val types = columnTypes(schema)
    val tokenBaseAliases = Set("TEXT", "TOKEN", "LOWER", "LENGTH")
    val hasBaseTokenMapping = Seq("TEXT", "TOKEN", "LOWER").exists(mapped.contains)
    attrs.foreach { attr =>
      expectedTypesByAttribute.get(attr).foreach { expected =>
        val mappedCol = getAttributeColumns.get(attr)
        val compatibleCols = types
          .collect {
            case (col, annotatorType) if expected.contains(annotatorType) => col
          }
          .toSeq
          .sorted

        require(
          mappedCol.isDefined || compatibleCols.nonEmpty,
          s"RuleBasedMatcher rule references attribute '$attr', but no compatible input column is available. " +
            s"Expected annotatorType ${expected.mkString(" or ")}. Add the column to inputCols and map it with setAttributeColumns if needed.")

        val attributeIsExplicit =
          mappedCol.isDefined || (tokenBaseAliases.contains(attr) && hasBaseTokenMapping)
        require(
          attributeIsExplicit || compatibleCols.length <= 1,
          s"RuleBasedMatcher rule references attribute '$attr', but multiple compatible input columns are available: " +
            s"${compatibleCols.mkString(", ")}. Map '$attr' explicitly with setAttributeColumns, for example Map('$attr' -> '${compatibleCols.head}').")
      }
    }
  }

  protected def validatePatternSafetyOrThrow(
      rules: Seq[com.johnsnowlabs.nlp.annotators.common.rulematcher.MatcherRule]): Unit = {
    rules.foreach { rule =>
      rule.patterns.zipWithIndex.foreach { case (pattern, patternIndex) =>
        val adjacentDangerousWildcards = pattern.elements
          .sliding(2)
          .exists(pair =>
            pair.forall(element => element.predicates.isEmpty && element.quantifier.max.isEmpty))
        require(
          !adjacentDangerousWildcards,
          s"RuleBasedMatcher rule '${rule.id}' pattern $patternIndex contains adjacent unbounded wildcard repetitions. " +
            "Merge them into a single wildcard repetition or use bounded quantifiers such as {0,5}.")
      }
    }
  }
}

/** Rule-based token matcher over multiple Spark NLP annotation columns.
  *
  * Patterns are supplied as JSON/JSONL rules. Each token pattern contains one or more attribute
  * predicates, and attributes are read from configured annotation columns such as TOKEN, POS,
  * NAMED_ENTITY, DEPENDENCY, or additional TOKEN columns like lemmas.
  */
class RuleBasedMatcher(override val uid: String)
    extends AnnotatorApproach[RuleBasedMatcherModel]
    with RuleBasedMatcherParams {

  override val description: String =
    "Matches token spans with declarative rules over multiple annotation attributes"

  override val outputAnnotatorType: AnnotatorType = CHUNK

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, TOKEN)

  val rules: Param[String] =
    new Param[String](this, "rules", "Inline JSON or JSONL rule definitions")

  val rulesResource: ExternalResourceParam =
    new ExternalResourceParam(this, "rulesResource", "External JSON or JSONL rule resource")

  def this() = this(Identifiable.randomUID("RULE_BASED_MATCHER"))

  def setRules(value: String): this.type = {
    require(get(rulesResource).isEmpty, "Only one of rules or rulesResource can be set")
    set(rules, value)
  }

  def setRulesResource(value: ExternalResource): this.type = {
    require(get(rules).isEmpty, "Only one of rules or rulesResource can be set")
    set(rulesResource, value)
  }

  def setRulesResource(
      path: String,
      readAs: ReadAs.Format = ReadAs.TEXT,
      options: Map[String, String] = Map("format" -> "text")): this.type =
    setRulesResource(ExternalResource(path, readAs, options))

  override protected def validate(schema: StructType): Boolean = validateInputColumns(schema)

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): RuleBasedMatcherModel = {
    validateInputColumnsOrThrow(dataset.schema)

    require(
      get(rules).nonEmpty || get(rulesResource).nonEmpty,
      "RuleBasedMatcher requires rules or rulesResource")

    val rawRules =
      if (get(rulesResource).nonEmpty) ResourceHelper.parseLines($(rulesResource)).mkString("\n")
      else $(rules)

    val normalizedRules = RulePatternParser.normalizeRulesJson(rawRules)
    val parsedRules = RulePatternParser.parseRules(normalizedRules)
    RuleMatcherEngine.validateRules(parsedRules)
    validateRuleAttributesOrThrow(parsedRules, dataset.schema)
    validatePatternSafetyOrThrow(parsedRules)

    val model = new RuleBasedMatcherModel()
      .setRulesJson(normalizedRules)
      .setInputCols(getInputCols)
      .setInputColumnTypes(columnTypes(dataset.schema))
      .setAttributeColumns(getAttributeColumns)
      .setAlignmentMode($(alignmentMode))
      .setOverlapStrategy($(overlapStrategy))

    if (isDefined(outputCol)) model.setOutputCol(getOutputCol)
    model
  }
}

object RuleBasedMatcher extends DefaultParamsReadable[RuleBasedMatcher]
