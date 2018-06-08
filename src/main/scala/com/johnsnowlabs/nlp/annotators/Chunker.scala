package com.johnsnowlabs.nlp.annotators

import scala.util.matching.Regex
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

class Chunker(override val uid: String) extends AnnotatorModel[Chunker] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  val regexParser: Param[String] = new Param(this, "regexParser",
                                        "A grammar based chunk parser")

  override val annotatorType: AnnotatorType = DOCUMENT
  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(POS)

  def setRegexParser(value: String): Chunker = set(regexParser, value)
  def getRegexParser: String = $(regexParser)

  def this() = this(Identifiable.randomUID("CHUNKER"))


  def patternMatchIndexes(pattern: Regex, text: String): List[(Int, Int)] =
    pattern.findAllMatchIn(text).map(index => (index.start, index.end )).toList

  def patternMatchFirstIndex(pattern: Regex, text: String): List[Int] =
    pattern.findAllMatchIn(text).map(_.start).toList

  def getIndexAnnotation(limits: (Int, Int), indexTags: List[(Int, Int)]): List[Int] = {
      val indexAnnotation = indexTags.zipWithIndex.collect{
        case (range, index) if limits._1-1 <= range._1 && limits._2 > range._2 => index
      }
    indexAnnotation
  }

  def getPhrase(indexAnnotation: List[Int], annotations: Seq[Annotation]): Seq[Annotation] = {
    val annotation = indexAnnotation.map(index => annotations.apply(index))
    annotation
  }

  private lazy val POSTagPattern: Regex = {
    val replacements = Map( "<" -> "(<", ">" -> ">)", "|" -> ">)|(<")
    replacements.foldLeft($(regexParser))((accumulatedParser, keyValueReplace) =>
      accumulatedParser.replaceAllLiterally(keyValueReplace._1, keyValueReplace._2)).r
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val POSFormatSentence = annotations.map(annotation => "<"+annotation.result+">")
                                        .mkString(" ").replaceAll("\\s","")
    val rangeMatches = patternMatchIndexes(POSTagPattern, POSFormatSentence)
    val indexLeftTags = patternMatchFirstIndex("<".r, POSFormatSentence)
    val indexRightTags = patternMatchFirstIndex(">".r, POSFormatSentence)
    val indexTags = indexLeftTags zip indexRightTags //merge two sequential collections
    val indexAnnotations = rangeMatches.map(range => getIndexAnnotation(range, indexTags))
    val chunkPhrases = indexAnnotations.map(indexAnnotation => getPhrase(indexAnnotation, annotations)).toArray

    val chunkAnnotations = chunkPhrases.zipWithIndex.map{ case (phrase, index) =>
      val result = phrase.map(annotation => annotation.metadata.head._2).mkString(" ")
      Annotation(annotatorType, 0, result.length, result, Map("chunk" -> index.toString))
    }

    chunkAnnotations

  }

}

object Chunker extends ParamsAndFeaturesReadable[Chunker]