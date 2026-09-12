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

package com.johnsnowlabs.nlp.annotators.common.rulematcher

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.Annotation
import org.json4s._
import org.json4s.jackson.JsonMethods.{compact, parse, render}

import java.util.regex.Pattern
import scala.collection.mutable.ArrayBuffer
import scala.util.{Failure, Success, Try}

case class MatcherRule(id: String, label: String, priority: Int, patterns: Seq[RulePattern])

case class RulePattern(elements: Seq[TokenPattern])

case class TokenPattern(predicates: Seq[AttributePredicate], quantifier: Quantifier)

case class Quantifier(op: String, min: Int, max: Option[Int])

sealed trait AttributePredicate extends Serializable {
  val attribute: String
}
case class EqualsPredicate(attribute: String, value: String) extends AttributePredicate
case class InPredicate(attribute: String, values: Set[String]) extends AttributePredicate
case class NotInPredicate(attribute: String, values: Set[String]) extends AttributePredicate
case class RegexPredicate(attribute: String, pattern: String) extends AttributePredicate {
  @transient lazy val compiledPattern: Pattern = Pattern.compile(pattern)
}
case class NotRegexPredicate(attribute: String, pattern: String) extends AttributePredicate {
  @transient lazy val compiledPattern: Pattern = Pattern.compile(pattern)
}
case class ExistsPredicate(attribute: String, expected: Boolean) extends AttributePredicate

case class TokenView(
    annotation: Annotation,
    documentKey: String,
    sentence: String,
    sentenceIndex: Int,
    documentIndex: Int,
    attributes: Map[String, String]) {

  def get(attribute: String): Option[String] =
    attributes.get(RuleMatcherUtils.normalize(attribute))
}

case class MatchCandidate(
    ruleId: String,
    label: String,
    priority: Int,
    patternIndex: Int,
    ruleIndex: Int,
    documentKey: String,
    sentence: String,
    sentenceTokenStart: Int,
    sentenceTokenEnd: Int,
    documentTokenStart: Int,
    documentTokenEnd: Int,
    begin: Int,
    end: Int,
    result: String)

private[rulematcher] object RuleMatcherUtils {
  def normalize(value: String): String = value.trim.toUpperCase

  def sentenceOf(annotation: Annotation): String =
    annotation.metadata.getOrElse("sentence", "0")

  def sourceColumnOf(annotation: Annotation): String =
    annotation.metadata.getOrElse("source_column", "unknown")

  def stripNerPrefix(tag: String): String = {
    val normalized = tag.trim
    if (normalized == "O" || normalized.isEmpty) normalized
    else {
      val dash = normalized.indexOf('-')
      if (dash > 0) {
        normalized.substring(0, dash) match {
          case "B" | "I" | "O" | "E" | "S" | "U" | "L" => normalized.substring(dash + 1)
          case _ => normalized
        }
      } else normalized
    }
  }
}

object AlignmentMode {
  val Strict = "STRICT"
  val Positional = "POSITIONAL"

  val values: Set[String] = Set(Strict, Positional)
}

object OverlapStrategy {
  val All = "ALL"
  val First = "FIRST"
  val Longest = "LONGEST"
  val PriorityLongest = "PRIORITY_LONGEST"

  val values: Set[String] = Set(All, First, Longest, PriorityLongest)
}

object RulePatternParser {

  def normalizeRulesJson(rawRules: String): String = {
    val trimmed = rawRules.trim
    require(trimmed.nonEmpty, "RuleBasedMatcher rules cannot be empty")

    val lines = trimmed
      .split("\\r?\\n")
      .map(_.trim)
      .filter(_.nonEmpty)
      .toList

    val looksLikeJsonLines =
      lines.length > 1 && lines.forall(line => line.startsWith("{") && line.endsWith("}"))

    val json = Try(parse(trimmed)) match {
      case Success(value) if !looksLikeJsonLines => value
      case Failure(jsonError) =>
        val objects = lines.zipWithIndex.map { case (line, index) =>
          Try(parse(line)) match {
            case Success(value) => value
            case Failure(jsonlError) =>
              throw new IllegalArgumentException(
                s"RuleBasedMatcher could not parse rules as JSON or JSONL. " +
                  s"JSON error: ${jsonError.getMessage}. JSONL line ${index + 1} error: ${jsonlError.getMessage}")
          }
        }
        JArray(objects)
      case Success(_) =>
        JArray(lines.zipWithIndex.map { case (line, index) =>
          Try(parse(line)) match {
            case Success(value) => value
            case Failure(jsonlError) =>
              throw new IllegalArgumentException(
                s"RuleBasedMatcher could not parse JSONL rule at line ${index + 1}: ${jsonlError.getMessage}")
          }
        })
    }

    json match {
      case array: JArray => compact(render(array))
      case obj: JObject => compact(render(JArray(List(obj))))
      case other =>
        throw new IllegalArgumentException(
          s"RuleBasedMatcher expects a JSON array, JSON object, or JSONL objects, got ${other.getClass.getSimpleName}")
    }
  }

  def parseRules(rawRules: String): Seq[MatcherRule] = {
    parse(normalizeRulesJson(rawRules)) match {
      case JArray(items) =>
        items.zipWithIndex.map { case (item, index) => parseRule(item, index) }
      case _ =>
        throw new IllegalArgumentException(
          "RuleBasedMatcher expected normalized rules as JSON array")
    }
  }

  private def parseRule(json: JValue, ruleIndex: Int): MatcherRule = {
    val obj = asObject(json, s"rule at index $ruleIndex")
    val id = stringField(obj, "id").getOrElse(s"rule_$ruleIndex")
    val label = stringField(obj, "label").orElse(stringField(obj, "entity")).getOrElse(id)
    val priority = intField(obj, "priority").getOrElse(0)

    val patterns = field(obj, "patterns") match {
      case JArray(items) if items.nonEmpty =>
        items.zipWithIndex.map { case (pattern, patternIndex) =>
          pattern match {
            case JArray(elements) if elements.nonEmpty =>
              RulePattern(elements.zipWithIndex.map { case (element, elementIndex) =>
                parseTokenPattern(element, id, patternIndex, elementIndex)
              })
            case _ =>
              throw new IllegalArgumentException(
                s"Rule '$id' pattern $patternIndex must be a non-empty JSON array")
          }
        }
      case _ =>
        throw new IllegalArgumentException(s"Rule '$id' must define a non-empty 'patterns' array")
    }

    MatcherRule(id, label, priority, patterns)
  }

  private def parseTokenPattern(
      json: JValue,
      ruleId: String,
      patternIndex: Int,
      elementIndex: Int): TokenPattern = {
    val context = s"rule '$ruleId' pattern $patternIndex token $elementIndex"
    val obj =
      asObject(json, context)
    val op = stringField(obj, "OP").getOrElse("")
    val predicates = obj.obj
      .filterNot { case (name, _) => name == "OP" }
      .flatMap { case (attribute, condition) =>
        parseCondition(attribute, condition, context)
      }
    TokenPattern(predicates, parseQuantifier(op, context))
  }

  def ruleAttributes(rules: Seq[MatcherRule]): Set[String] =
    rules.flatMap(_.patterns).flatMap(_.elements).flatMap(_.predicates.map(_.attribute)).toSet

  private def parseCondition(
      attribute: String,
      condition: JValue,
      context: String): Seq[AttributePredicate] = {
    condition match {
      case JString(value) => Seq(EqualsPredicate(attribute, value))
      case JInt(value) => Seq(EqualsPredicate(attribute, value.toString))
      case JDouble(value) => Seq(EqualsPredicate(attribute, value.toString))
      case JDecimal(value) => Seq(EqualsPredicate(attribute, value.toString))
      case JBool(value) => Seq(EqualsPredicate(attribute, value.toString))
      case JObject(fields) if fields.nonEmpty =>
        fields.flatMap { case (operation, value) =>
          operation.toUpperCase match {
            case "IN" =>
              val predicateValues = values(value, operation)
              require(
                predicateValues.nonEmpty,
                s"RuleBasedMatcher predicate '$operation' for '$attribute' in $context cannot be empty")
              Seq(InPredicate(attribute, predicateValues.toSet))
            case "NOT_IN" =>
              val predicateValues = values(value, operation)
              require(
                predicateValues.nonEmpty,
                s"RuleBasedMatcher predicate '$operation' for '$attribute' in $context cannot be empty")
              Seq(NotInPredicate(attribute, predicateValues.toSet))
            case "REGEX" =>
              val regex = stringValue(value, operation)
              require(
                regex.nonEmpty,
                s"RuleBasedMatcher predicate '$operation' for '$attribute' in $context cannot be empty")
              Seq(RegexPredicate(attribute, regex))
            case "NOT_REGEX" =>
              val regex = stringValue(value, operation)
              require(
                regex.nonEmpty,
                s"RuleBasedMatcher predicate '$operation' for '$attribute' in $context cannot be empty")
              Seq(NotRegexPredicate(attribute, regex))
            case "EXISTS" => Seq(ExistsPredicate(attribute, booleanValue(value, operation)))
            case unknown =>
              throw new IllegalArgumentException(
                s"Unsupported RuleBasedMatcher predicate '$unknown' for attribute '$attribute' in $context")
          }
        }
      case _ =>
        throw new IllegalArgumentException(
          s"Unsupported RuleBasedMatcher condition for attribute '$attribute' in $context: ${compact(
              render(condition))}")
    }
  }

  private def parseQuantifier(op: String, context: String): Quantifier = {
    val trimmed = op.trim
    trimmed match {
      case "" => Quantifier(trimmed, min = 1, max = Some(1))
      case "?" => Quantifier(trimmed, min = 0, max = Some(1))
      case "*" => Quantifier(trimmed, min = 0, max = None)
      case "+" => Quantifier(trimmed, min = 1, max = None)
      case _ if trimmed.matches("^\\{\\d+\\}$") =>
        val count = trimmed.substring(1, trimmed.length - 1).toInt
        Quantifier(trimmed, min = count, max = Some(count))
      case _ if trimmed.matches("^\\{\\d+,\\d*\\}$") =>
        val parts = trimmed.substring(1, trimmed.length - 1).split(",", -1)
        val min = parts(0).toInt
        val max = if (parts(1).isEmpty) None else Some(parts(1).toInt)
        require(
          max.forall(_ >= min),
          s"Invalid RuleBasedMatcher quantifier '$op' in $context: max must be >= min")
        Quantifier(trimmed, min, max)
      case _ =>
        throw new IllegalArgumentException(
          s"Unsupported RuleBasedMatcher quantifier '$op' in $context. Use '', '?', '*', '+', '{n}', or '{n,m}'.")
    }
  }

  private def field(obj: JObject, name: String): JValue =
    obj.obj.find { case (fieldName, _) => fieldName == name }.map(_._2).getOrElse(JNothing)

  private def stringField(obj: JObject, name: String): Option[String] =
    field(obj, name) match {
      case JString(value) => Some(value)
      case JNothing => None
      case other =>
        throw new IllegalArgumentException(
          s"Field '$name' must be a string, got ${compact(render(other))}")
    }

  private def intField(obj: JObject, name: String): Option[Int] =
    field(obj, name) match {
      case JInt(value) => Some(value.toInt)
      case JNothing => None
      case other =>
        throw new IllegalArgumentException(
          s"Field '$name' must be an integer, got ${compact(render(other))}")
    }

  private def asObject(json: JValue, context: String): JObject =
    json match {
      case obj: JObject => obj
      case _ => throw new IllegalArgumentException(s"Expected JSON object for $context")
    }

  private def values(json: JValue, operation: String): Seq[String] =
    json match {
      case JArray(items) => items.map(stringValue(_, operation))
      case other => Seq(stringValue(other, operation))
    }

  private def stringValue(json: JValue, operation: String): String =
    json match {
      case JString(value) => value
      case JInt(value) => value.toString
      case JDouble(value) => value.toString
      case JDecimal(value) => value.toString
      case JBool(value) => value.toString
      case _ =>
        throw new IllegalArgumentException(
          s"Predicate '$operation' expects a scalar string/number/boolean value")
    }

  private def booleanValue(json: JValue, operation: String): Boolean =
    json match {
      case JBool(value) => value
      case JString(value) => value.toBoolean
      case _ => throw new IllegalArgumentException(s"Predicate '$operation' expects a boolean")
    }
}

object AnnotationAligner {

  private case class DocumentRef(
      key: String,
      annotation: Annotation,
      sourceOrder: Int,
      annotationOrder: Int)

  private case class AnnotationWithDocument(
      annotation: Annotation,
      documentKey: String,
      sourceIndex: Int)

  def annotationsByColumn(
      annotations: Seq[Annotation],
      inputCols: Array[String],
      columnTypes: Map[String, String]): Map[String, Seq[Annotation]] = {
    val withSource = annotations.filter(_.metadata.contains("source_column"))
    if (withSource.nonEmpty) {
      inputCols.map { col =>
        col -> withSource.filter(_.metadata.get("source_column").contains(col))
      }.toMap
    } else {
      splitFlattenedSafely(annotations, inputCols, columnTypes)
    }
  }

  def buildTokenViews(
      annotationsByColumn: Map[String, Seq[Annotation]],
      inputCols: Array[String],
      columnTypes: Map[String, String],
      attributeColumns: Map[String, String],
      alignmentMode: String): Seq[Seq[TokenView]] = {
    val effectiveAttributeColumns =
      inferAttributeColumns(inputCols, columnTypes) ++ normalizeAttributeColumns(attributeColumns)

    val baseTokenColumn = baseTokenCol(inputCols, columnTypes, effectiveAttributeColumns)
    val documents = documentRefs(annotationsByColumn, inputCols)
    val tokens = assignDocumentKeys(
      annotationsByColumn
        .getOrElse(baseTokenColumn, Seq.empty),
      documents)
    val sortedTokens = tokens
      .sortBy(token =>
        (
          token.documentKey,
          RuleMatcherUtils.sentenceOf(token.annotation),
          token.annotation.begin,
          token.annotation.end,
          token.sourceIndex))

    val documentKeysByAnnotationColumn: Map[String, Seq[AnnotationWithDocument]] =
      effectiveAttributeColumns.values.toSeq.distinct.map { col =>
        col -> assignDocumentKeys(annotationsByColumn.getOrElse(col, Seq.empty), documents)
      }.toMap

    val alignedByAttribute: Map[String, Map[(String, String, Int, Int), Annotation]] =
      effectiveAttributeColumns
        .filterNot { case (_, col) => col == baseTokenColumn }
        .map { case (attribute, col) =>
          attribute -> documentKeysByAnnotationColumn
            .getOrElse(col, Seq.empty)
            .map { annotationWithDocument =>
              val annotation = annotationWithDocument.annotation
              (
                annotationWithDocument.documentKey,
                RuleMatcherUtils.sentenceOf(annotation),
                annotation.begin,
                annotation.end) -> annotation
            }
            .toMap
        }

    val positionalByAttribute: Map[String, Map[(String, String, Int), Annotation]] =
      if (alignmentMode == AlignmentMode.Positional) {
        effectiveAttributeColumns
          .filterNot { case (_, col) => col == baseTokenColumn }
          .map { case (attribute, col) =>
            val indexed = documentKeysByAnnotationColumn
              .getOrElse(col, Seq.empty)
              .groupBy(annotationWithDocument =>
                (
                  annotationWithDocument.documentKey,
                  RuleMatcherUtils.sentenceOf(annotationWithDocument.annotation)))
              .flatMap { case ((documentKey, sentence), anns) =>
                anns
                  .sortBy(a => (a.annotation.begin, a.annotation.end, a.sourceIndex))
                  .zipWithIndex
                  .map { case (annotationWithDocument, idx) =>
                    (documentKey, sentence, idx) -> annotationWithDocument.annotation
                  }
              }
            attribute -> indexed
          }
      } else Map.empty

    val documentIndexByTokenSourceIndex = sortedTokens
      .groupBy(_.documentKey)
      .values
      .flatMap { documentTokens =>
        documentTokens
          .sortBy(t =>
            (
              sentenceSortKey(RuleMatcherUtils.sentenceOf(t.annotation)),
              t.annotation.begin,
              t.annotation.end,
              t.sourceIndex))
          .zipWithIndex
          .map { case (token, documentIndex) => token.sourceIndex -> documentIndex }
      }
      .toMap

    sortedTokens
      .groupBy(token => (token.documentKey, RuleMatcherUtils.sentenceOf(token.annotation)))
      .toSeq
      .sortBy { case ((documentKey, sentence), _) => (documentKey, sentenceSortKey(sentence)) }
      .map { case ((documentKey, sentence), sentenceTokens) =>
        sentenceTokens
          .sortBy(t => (t.annotation.begin, t.annotation.end, t.sourceIndex))
          .zipWithIndex
          .map { case (tokenWithDocument, index) =>
            val token = tokenWithDocument.annotation
            val baseAttributes = baseTokenAttributes(token)
            val mappedAttributes = effectiveAttributeColumns.flatMap { case (attribute, col) =>
              if (col == baseTokenColumn) {
                valueForAttribute(attribute, token).map(value =>
                  RuleMatcherUtils.normalize(attribute) -> value)
              } else {
                val exact = alignedByAttribute
                  .getOrElse(attribute, Map.empty)
                  .get((documentKey, sentence, token.begin, token.end))
                val positional = positionalByAttribute
                  .getOrElse(attribute, Map.empty)
                  .get((documentKey, sentence, index))
                exact.orElse(positional).flatMap { annotation =>
                  valueForAttribute(attribute, annotation).map(value =>
                    RuleMatcherUtils.normalize(attribute) -> value)
                }
              }
            }
            TokenView(
              token,
              documentKey,
              sentence,
              index,
              documentIndexByTokenSourceIndex.getOrElse(tokenWithDocument.sourceIndex, index),
              baseAttributes ++ mappedAttributes)
          }
      }
  }

  private def splitFlattenedSafely(
      annotations: Seq[Annotation],
      inputCols: Array[String],
      columnTypes: Map[String, String]): Map[String, Seq[Annotation]] = {
    val duplicateTypes = inputCols
      .groupBy(col => columnTypes.getOrElse(col, ""))
      .filter { case (_, cols) => cols.length > 1 }
    require(
      duplicateTypes.isEmpty,
      "RuleBasedMatcher cannot safely infer source columns from flattened annotations when " +
        s"multiple input columns share an annotator type: ${duplicateTypes
            .map { case (annotatorType, cols) =>
              s"$annotatorType -> ${cols.mkString("[", ", ", "]")}"
            }
            .mkString(", ")}. Use DataFrame/Pipeline/LightPipeline execution or provide annotations with source_column metadata.")

    inputCols.map { col =>
      val annotatorType = columnTypes.getOrElse(col, "")
      col -> annotations.filter(_.annotatorType == annotatorType)
    }.toMap
  }

  private def documentRefs(
      annotationsByColumn: Map[String, Seq[Annotation]],
      inputCols: Array[String]): Seq[DocumentRef] = {
    val inputOrder = inputCols.zipWithIndex.toMap
    inputCols
      .flatMap { col =>
        annotationsByColumn
          .getOrElse(col, Seq.empty)
          .zipWithIndex
          .filter { case (annotation, _) => annotation.annotatorType == DOCUMENT }
          .map { case (annotation, index) =>
            val key = Seq(
              col,
              index.toString,
              annotation.begin.toString,
              annotation.end.toString,
              RuleMatcherUtils.sentenceOf(annotation)).mkString(":")
            DocumentRef(key, annotation, inputOrder.getOrElse(col, Int.MaxValue), index)
          }
      }
      .sortBy(ref => (ref.sourceOrder, ref.annotationOrder))
  }

  private def assignDocumentKeys(
      annotations: Seq[Annotation],
      documents: Seq[DocumentRef]): Seq[AnnotationWithDocument] = {
    if (annotations.isEmpty) return Seq.empty

    val orderedDocuments = documents.sortBy(ref => (ref.sourceOrder, ref.annotationOrder))
    var currentDocumentIndex = 0
    var previous: Option[Annotation] = None

    annotations.zipWithIndex.map { case (annotation, sourceIndex) =>
      val candidates = candidateDocuments(annotation, orderedDocuments)
      val indexedCandidates = candidates.map(ref => ref -> orderedDocuments.indexOf(ref))

      if (isDocumentReset(
          previous,
          annotation) && currentDocumentIndex < orderedDocuments.length - 1) {
        indexedCandidates
          .find { case (_, index) => index > currentDocumentIndex }
          .map { case (_, index) => currentDocumentIndex = index }
          .getOrElse(currentDocumentIndex += 1)
      }

      val selected = indexedCandidates
        .find { case (_, index) => index == currentDocumentIndex }
        .orElse(indexedCandidates.find { case (_, index) => index > currentDocumentIndex })
        .orElse(indexedCandidates.headOption)

      selected.foreach { case (_, index) => currentDocumentIndex = index }
      previous = Some(annotation)
      AnnotationWithDocument(
        annotation,
        selected.map(_._1.key).getOrElse(fallbackDocumentKey(annotation)),
        sourceIndex)
    }
  }

  private def candidateDocuments(
      annotation: Annotation,
      documents: Seq[DocumentRef]): Seq[DocumentRef] =
    documents.filter(ref =>
      annotation.begin >= ref.annotation.begin && annotation.end <= ref.annotation.end)

  private def isDocumentReset(previous: Option[Annotation], current: Annotation): Boolean =
    previous.exists { prev =>
      val prevSentence = sentenceSortKey(RuleMatcherUtils.sentenceOf(prev))
      val currentSentence = sentenceSortKey(RuleMatcherUtils.sentenceOf(current))
      currentSentence < prevSentence ||
      (currentSentence == prevSentence && current.begin < prev.begin)
    }

  private def fallbackDocumentKey(annotation: Annotation): String = {
    Seq(
      "no_document",
      RuleMatcherUtils.sourceColumnOf(annotation),
      RuleMatcherUtils.sentenceOf(annotation)).mkString(":")
  }

  private def normalizeAttributeColumns(
      attributeColumns: Map[String, String]): Map[String, String] =
    attributeColumns.map { case (attribute, col) => RuleMatcherUtils.normalize(attribute) -> col }

  private def inferAttributeColumns(
      inputCols: Array[String],
      columnTypes: Map[String, String]): Map[String, String] = {
    val byType = inputCols.groupBy(col => columnTypes.getOrElse(col, ""))
    val tokenColumns = byType.getOrElse(TOKEN, Array.empty)
    val inferred = scala.collection.mutable.Map.empty[String, String]

    tokenColumns.headOption.foreach { col =>
      inferred += "TEXT" -> col
      inferred += "TOKEN" -> col
      inferred += "LOWER" -> col
      inferred += "LENGTH" -> col
    }
    tokenColumns.find(_.toLowerCase.contains("lemma")).foreach(col => inferred += "LEMMA" -> col)
    byType.get(POS).flatMap(_.headOption).foreach(col => inferred += "POS" -> col)
    byType.get(NAMED_ENTITY).flatMap(_.headOption).foreach { col =>
      inferred += "NER" -> col
      inferred += "NER_TAG" -> col
      inferred += "NER_TYPE" -> col
      inferred += "ENT_TYPE" -> col
    }
    byType.get(DEPENDENCY).flatMap(_.headOption).foreach { col =>
      inferred += "DEP" -> col
      inferred += "HEAD" -> col
      inferred += "HEAD_BEGIN" -> col
      inferred += "HEAD_END" -> col
    }
    byType.get(LABELED_DEPENDENCY).flatMap(_.headOption).foreach { col =>
      inferred += "DEP_LABEL" -> col
      inferred += "RELATION" -> col
    }

    inferred.toMap
  }

  private def baseTokenCol(
      inputCols: Array[String],
      columnTypes: Map[String, String],
      attributeColumns: Map[String, String]): String = {
    attributeColumns
      .get("TEXT")
      .orElse(attributeColumns.get("TOKEN"))
      .orElse(attributeColumns.get("LOWER"))
      .orElse(inputCols.find(col => columnTypes.getOrElse(col, "") == TOKEN))
      .getOrElse {
        throw new IllegalArgumentException(
          "RuleBasedMatcher requires a TOKEN input column for base token alignment")
      }
  }

  private def baseTokenAttributes(token: Annotation): Map[String, String] = {
    val metadataAttrs = token.metadata.map { case (key, value) =>
      RuleMatcherUtils.normalize(s"META.$key") -> value
    }.toMap
    metadataAttrs ++ Map(
      "TEXT" -> token.result,
      "TOKEN" -> token.result,
      "LOWER" -> token.result.toLowerCase,
      "LENGTH" -> token.result.length.toString)
  }

  private def sentenceSortKey(sentence: String): Int =
    try sentence.toInt
    catch {
      case _: NumberFormatException => Int.MaxValue
    }

  private def valueForAttribute(attribute: String, annotation: Annotation): Option[String] = {
    val normalized = RuleMatcherUtils.normalize(attribute)
    if (normalized.startsWith("META.")) {
      val metadataKey = attribute.dropWhile(_ != '.').drop(1)
      annotation.metadata.get(metadataKey).orElse {
        annotation.metadata
          .find { case (key, _) =>
            RuleMatcherUtils.normalize(s"META.$key") == normalized
          }
          .map(_._2)
      }
    } else {
      normalized match {
        case "LOWER" => Some(annotation.result.toLowerCase)
        case "LENGTH" => Some(annotation.result.length.toString)
        case "NER" => Some(annotation.result)
        case "NER_TAG" => Some(annotation.result)
        case "NER_TYPE" => Some(RuleMatcherUtils.stripNerPrefix(annotation.result))
        case "ENT_TYPE" => Some(RuleMatcherUtils.stripNerPrefix(annotation.result))
        case "HEAD" => annotation.metadata.get("head").orElse(Some(annotation.result))
        case "HEAD_BEGIN" => annotation.metadata.get("head.begin")
        case "HEAD_END" => annotation.metadata.get("head.end")
        case _ => Some(annotation.result)
      }
    }
  }
}

object RuleMatcherEngine {

  def findMatches(
      tokensBySentence: Seq[Seq[TokenView]],
      rules: Seq[MatcherRule],
      overlapStrategy: String): Seq[MatchCandidate] = {
    val candidates = rules.zipWithIndex.flatMap { case (rule, ruleIndex) =>
      rule.patterns.zipWithIndex.flatMap { case (pattern, patternIndex) =>
        tokensBySentence.flatMap { sentenceTokens =>
          findPatternMatches(sentenceTokens, rule, ruleIndex, pattern, patternIndex)
        }
      }
    }

    resolveOverlaps(candidates, overlapStrategy)
  }

  def validateRules(rules: Seq[MatcherRule]): Unit = {
    rules.foreach { rule =>
      rule.patterns.zipWithIndex.foreach { case (pattern, patternIndex) =>
        pattern.elements.foreach { tokenPattern =>
          tokenPattern.predicates.foreach {
            case predicate: RegexPredicate =>
              try predicate.compiledPattern
              catch {
                case exception: Exception =>
                  throw new IllegalArgumentException(
                    s"Invalid RuleBasedMatcher REGEX predicate for attribute '${predicate.attribute}' in rule '${rule.id}' pattern $patternIndex: ${exception.getMessage}",
                    exception)
              }
            case predicate: NotRegexPredicate =>
              try predicate.compiledPattern
              catch {
                case exception: Exception =>
                  throw new IllegalArgumentException(
                    s"Invalid RuleBasedMatcher NOT_REGEX predicate for attribute '${predicate.attribute}' in rule '${rule.id}' pattern $patternIndex: ${exception.getMessage}",
                    exception)
              }
            case _ =>
          }
        }
      }
    }
  }

  private def findPatternMatches(
      sentenceTokens: Seq[TokenView],
      rule: MatcherRule,
      ruleIndex: Int,
      pattern: RulePattern,
      patternIndex: Int): Seq[MatchCandidate] = {
    if (sentenceTokens.isEmpty) return Seq.empty

    val memo = scala.collection.mutable.Map.empty[(Int, Int), Seq[Int]]
    sentenceTokens.indices.flatMap { start =>
      val endPositions =
        matchFrom(sentenceTokens, pattern.elements, elementIndex = 0, tokenIndex = start, memo)
      endPositions.distinct.filter(_ > start).map { endExclusive =>
        val matchedTokens = sentenceTokens.slice(start, endExclusive)
        val begin = matchedTokens.head.annotation.begin
        val end = matchedTokens.last.annotation.end
        MatchCandidate(
          ruleId = rule.id,
          label = rule.label,
          priority = rule.priority,
          patternIndex = patternIndex,
          ruleIndex = ruleIndex,
          documentKey = matchedTokens.head.documentKey,
          sentence = matchedTokens.head.sentence,
          sentenceTokenStart = matchedTokens.head.sentenceIndex,
          sentenceTokenEnd = matchedTokens.last.sentenceIndex,
          documentTokenStart = matchedTokens.head.documentIndex,
          documentTokenEnd = matchedTokens.last.documentIndex,
          begin = begin,
          end = end,
          result = matchedTokens.map(_.annotation.result).mkString(" "))
      }
    }
  }

  private def matchFrom(
      tokens: Seq[TokenView],
      elements: Seq[TokenPattern],
      elementIndex: Int,
      tokenIndex: Int,
      memo: scala.collection.mutable.Map[(Int, Int), Seq[Int]]): Seq[Int] = {
    if (elementIndex == elements.length) return Seq(tokenIndex)
    if (tokenIndex > tokens.length) return Seq.empty
    memo.get((elementIndex, tokenIndex)) match {
      case Some(result) => return result
      case None =>
    }

    val element = elements(elementIndex)
    val maxCount = element.quantifier.max.getOrElse(tokens.length - tokenIndex)
    val cappedMax = Math.min(maxCount, tokens.length - tokenIndex)
    val validCounts = ArrayBuffer.empty[Int]
    var count = 0
    var stillMatches = true

    if (element.quantifier.min == 0) validCounts += 0

    while (count < cappedMax && stillMatches) {
      val nextToken = tokens(tokenIndex + count)
      stillMatches = tokenMatches(nextToken, element.predicates)
      if (stillMatches) {
        count += 1
        if (count >= element.quantifier.min) validCounts += count
      }
    }

    val countsToTry =
      if (elementIndex == elements.length - 1) validCounts.lastOption.toSeq
      else validCounts.toSeq.reverse

    val result = countsToTry.flatMap { matchedCount =>
      matchFrom(tokens, elements, elementIndex + 1, tokenIndex + matchedCount, memo)
    }
    memo((elementIndex, tokenIndex)) = result
    result
  }

  private def tokenMatches(token: TokenView, predicates: Seq[AttributePredicate]): Boolean =
    predicates.forall {
      case EqualsPredicate(attribute, value) => token.get(attribute).contains(value)
      case InPredicate(attribute, values) => token.get(attribute).exists(values.contains)
      case NotInPredicate(attribute, values) =>
        token.get(attribute).exists(v => !values.contains(v))
      case predicate: RegexPredicate =>
        token
          .get(predicate.attribute)
          .exists(value => predicate.compiledPattern.matcher(value).find())
      case predicate: NotRegexPredicate =>
        token
          .get(predicate.attribute)
          .exists(value => !predicate.compiledPattern.matcher(value).find())
      case ExistsPredicate(attribute, expected) => token.get(attribute).isDefined == expected
    }

  private def resolveOverlaps(
      candidates: Seq[MatchCandidate],
      overlapStrategy: String): Seq[MatchCandidate] = {
    val normalized = overlapStrategy.toUpperCase
    normalized match {
      case OverlapStrategy.All =>
        candidates.sortBy(c => (c.documentKey, c.begin, c.end, c.ruleIndex, c.patternIndex))
      case OverlapStrategy.First =>
        greedySelect(
          candidates.sortBy(c => (c.documentKey, c.begin, c.ruleIndex, c.patternIndex)))
      case OverlapStrategy.Longest =>
        greedySelect(candidates.sortBy(c =>
          (c.documentKey, -(c.end - c.begin), c.begin, c.ruleIndex, c.patternIndex)))
      case OverlapStrategy.PriorityLongest =>
        greedySelect(candidates.sortBy(c =>
          (c.documentKey, -c.priority, -(c.end - c.begin), c.begin, c.ruleIndex)))
          .sortBy(c => (c.documentKey, c.begin, c.end, c.ruleIndex))
      case other =>
        throw new IllegalArgumentException(
          s"Unsupported overlap strategy '$other'. Use ${OverlapStrategy.values.mkString(", ")}")
    }
  }

  private def greedySelect(candidates: Seq[MatchCandidate]): Seq[MatchCandidate] = {
    val selected = ArrayBuffer.empty[MatchCandidate]
    candidates.foreach { candidate =>
      if (!selected.exists(overlaps(_, candidate))) selected += candidate
    }
    selected.toSeq.sortBy(c => (c.documentKey, c.begin, c.end, c.ruleIndex, c.patternIndex))
  }

  private def overlaps(left: MatchCandidate, right: MatchCandidate): Boolean =
    left.documentKey == right.documentKey && left.begin <= right.end && right.begin <= left.end
}
