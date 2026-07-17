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
  AnnotationAligner,
  MatchCandidate,
  RuleMatcherEngine,
  RulePatternParser
}
import com.johnsnowlabs.nlp.serialization.StructJSONFeature
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.param.StringArrayParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.StructType

class RuleBasedMatcherModel(override val uid: String)
    extends AnnotatorModel[RuleBasedMatcherModel]
    with HasSimpleAnnotate[RuleBasedMatcherModel]
    with HasLightPipelineAnnotate
    with RuleBasedMatcherParams {

  override val outputAnnotatorType: AnnotatorType = CHUNK

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, TOKEN)

  val inputColumnTypes: StringArrayParam = new StringArrayParam(
    this,
    "inputColumnTypes",
    "Input column annotator types encoded as column=annotatorType")

  val rulesJson: StructJSONFeature[String] =
    new StructJSONFeature[String](this, "rulesJson")(identity, identity)

  setDefault(inputColumnTypes -> Array.empty[String])

  def this() = this(Identifiable.randomUID("RULE_BASED_MATCHER"))

  def setRulesJson(value: String): this.type = {
    RuleMatcherEngine.validateRules(RulePatternParser.parseRules(value))
    set(rulesJson, value)
  }

  def getRulesJson: String = $$(rulesJson)

  def setInputColumnTypes(value: Map[String, String]): this.type =
    set(
      inputColumnTypes,
      value.map { case (col, annotatorType) => s"$col=$annotatorType" }.toArray)

  def setInputColumnTypes(value: Array[String]): this.type = set(inputColumnTypes, value)

  def getInputColumnTypes: Map[String, String] =
    parseKeyValueEntries($(inputColumnTypes), "inputColumnTypes")

  override protected def validate(schema: StructType): Boolean = {
    validateInputColumnsOrThrow(schema)
    validatePersistedColumnTypes(schema)
    true
  }

  override protected def extraValidateMsg: String =
    "RuleBasedMatcher input columns must be annotation columns and attributeColumns must reference inputCols"

  override protected def extraValidate(structType: StructType): Boolean = {
    validateInputColumnsOrThrow(structType)
    validatePersistedColumnTypes(structType)
    true
  }

  override def beforeAnnotateLight(
      annotations: Map[String, Seq[IAnnotation]],
      metadata: Map[String, Seq[String]]): Map[String, Seq[IAnnotation]] = {
    getInputCols.foldLeft(annotations) { case (updated, colName) =>
      updated.get(colName) match {
        case Some(colAnnotations) =>
          updated.updated(
            colName,
            colAnnotations.map {
              case annotation: Annotation =>
                annotation.copy(metadata = annotation.metadata + ("source_column" -> colName))
              case other => other
            })
        case None => updated
      }
    }
  }

  @transient private lazy val parsedRules = RulePatternParser.parseRules($$(rulesJson))

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val annotationsByColumn =
      AnnotationAligner.annotationsByColumn(annotations, getInputCols, getInputColumnTypes)
    annotateByColumn(annotationsByColumn)
  }

  override def dfAnnotate: UserDefinedFunction = {
    val inputCols = getInputCols
    udf { annotationProperties: Seq[AnnotationContent] =>
      val annotationsByColumn = annotationProperties
        .zip(inputCols)
        .map { case (annotationRows, colName) =>
          val annotations = annotationRows.map { row =>
            val annotation = Annotation(row)
            annotation.copy(metadata = annotation.metadata + ("source_column" -> colName))
          }
          colName -> annotations
        }
        .toMap
      annotateByColumn(annotationsByColumn)
    }
  }

  private def annotateByColumn(
      annotationsByColumn: Map[String, Seq[Annotation]]): Seq[Annotation] = {
    val effectiveColumnTypes =
      if (getInputColumnTypes.nonEmpty) getInputColumnTypes
      else
        annotationsByColumn.flatMap { case (col, annotations) =>
          annotations.headOption.map(annotation => col -> annotation.annotatorType)
        }

    val tokensBySentence = AnnotationAligner.buildTokenViews(
      annotationsByColumn,
      getInputCols,
      effectiveColumnTypes,
      getAttributeColumns,
      $(alignmentMode))

    val documents = annotationsByColumn.values.flatten
      .filter(_.annotatorType == DOCUMENT)
      .toSeq

    RuleMatcherEngine
      .findMatches(tokensBySentence, parsedRules, $(overlapStrategy))
      .zipWithIndex
      .map { case (candidate, chunkIndex) =>
        Annotation(
          outputAnnotatorType,
          candidate.begin,
          candidate.end,
          chunkResult(candidate, annotationsByColumn, documents),
          metadata(candidate, chunkIndex))
      }
  }

  private def chunkResult(
      candidate: MatchCandidate,
      annotationsByColumn: Map[String, Seq[Annotation]],
      documents: Seq[Annotation]): String = {
    documentByKey(candidate.documentKey, annotationsByColumn)
      .orElse(
        documents.find(doc =>
          doc.metadata.getOrElse("sentence", "0") == candidate.sentence &&
            candidate.begin >= doc.begin &&
            candidate.end <= doc.end))
      .orElse(documents.find(doc =>
        candidate.begin >= doc.begin &&
          candidate.end <= doc.end))
      .flatMap { doc =>
        val start = candidate.begin - doc.begin
        val endExclusive = candidate.end - doc.begin + 1
        if (start >= 0 && endExclusive <= doc.result.length && start < endExclusive)
          Some(doc.result.substring(start, endExclusive))
        else None
      }
      .getOrElse(candidate.result)
  }

  private def documentByKey(
      documentKey: String,
      annotationsByColumn: Map[String, Seq[Annotation]]): Option[Annotation] = {
    val parts = documentKey.split(":", 5)
    if (parts.length != 5) None
    else {
      val col = parts(0)
      val index = scala.util.Try(parts(1).toInt).toOption
      index.flatMap(idx =>
        annotationsByColumn.get(col).flatMap(_.lift(idx).filter(_.annotatorType == DOCUMENT)))
    }
  }

  private def metadata(candidate: MatchCandidate, chunkIndex: Int): Map[String, String] =
    Map(
      "entity" -> candidate.label,
      "label" -> candidate.label,
      "rule" -> candidate.ruleId,
      "priority" -> candidate.priority.toString,
      "pattern" -> candidate.patternIndex.toString,
      "document" -> candidate.documentKey,
      "documentKey" -> candidate.documentKey,
      "sentence" -> candidate.sentence,
      "chunk" -> chunkIndex.toString,
      "tokenBegin" -> candidate.sentenceTokenStart.toString,
      "tokenEnd" -> candidate.sentenceTokenEnd.toString,
      "sentenceTokenBegin" -> candidate.sentenceTokenStart.toString,
      "sentenceTokenEnd" -> candidate.sentenceTokenEnd.toString,
      "documentTokenBegin" -> candidate.documentTokenStart.toString,
      "documentTokenEnd" -> candidate.documentTokenEnd.toString)

  private def validatePersistedColumnTypes(schema: StructType): Unit = {
    val persistedTypes = getInputColumnTypes
    if (persistedTypes.nonEmpty) {
      val currentTypes = columnTypes(schema)
      val mismatches = persistedTypes.flatMap { case (col, expectedType) =>
        currentTypes
          .get(col)
          .filter(_ != expectedType)
          .map(actualType => s"$col expected $expectedType but found $actualType")
      }
      require(
        mismatches.isEmpty,
        s"RuleBasedMatcherModel input column annotator types differ from the fitted model: ${mismatches
            .mkString(", ")}")
    }
  }
}

object RuleBasedMatcherModel extends ParamsAndFeaturesReadable[RuleBasedMatcherModel]
