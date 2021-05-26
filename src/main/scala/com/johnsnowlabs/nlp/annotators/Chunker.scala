package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, HasSimpleAnnotate}
import org.apache.spark.ml.param.StringArrayParam
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.util.matching.Regex


/**
  * This annotator matches a pattern of part-of-speech tags in order to return meaningful phrases from document
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ChunkerTestSpec.scala]] for reference on how to use this API.
  *
  * @param uid internal uid required to generate writable annotators
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  **/
class Chunker(override val uid: String) extends AnnotatorModel[Chunker] with HasSimpleAnnotate[Chunker] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** an array of grammar based chunk parsers
    *
    * @group param
    **/
  val regexParsers = new StringArrayParam(this, "regexParsers", "an array of grammar based chunk parsers")

  /** Output annotator type : CHUNK
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = CHUNK
  /** Input annotator type : DOCUMENT, POS
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, POS)

  /** A list of regex patterns to match chunks, for example: Array(“‹DT›?‹JJ›*‹NN›”)
    *
    * @group setParam
    **/
  def setRegexParsers(value: Array[String]): Chunker = set(regexParsers, value)

  /** adds a pattern to the current list of chunk patterns, for example: “‹DT›?‹JJ›*‹NN›”
    *
    * @group setParam
    **/
  def addRegexParser(value: String): Chunker = {
    set(regexParsers, get(regexParsers).getOrElse(Array.empty[String]) :+ value)
  }

  /** A list of regex patterns to match chunks, for example: Array(“‹DT›?‹JJ›*‹NN›”)
    *
    * @group getParam
    **/
  def getRegexParsers: Array[String] = $(regexParsers)

  def this() = this(Identifiable.randomUID("CHUNKER"))

  /** @group param */
  private lazy val replacements = Map("<" -> "(?:<", ">" -> ">)", "|" -> ">|<")
  /** @group param */
  private lazy val POSTagPatterns: Array[Regex] = {
    getRegexParsers.map(regexParser => replaceRegexParser(regexParser))
  }

  private def replaceRegexParser(regexParser: String): Regex = {
    replacements.foldLeft(regexParser)((accumulatedParser, keyValueReplace) =>
      accumulatedParser.replaceAllLiterally(keyValueReplace._1, keyValueReplace._2)).r
  }

  private def patternMatchIndexes(pattern: Regex, text: String): List[(Int, Int)] = {
    pattern.findAllMatchIn(text).map(index => (index.start, index.end )).toList
  }

  private def patternMatchFirstIndex(pattern: Regex, text: String): List[Int] =
    pattern.findAllMatchIn(text).map(_.start).toList

  private def getIndexAnnotation(limits: (Int, Int), indexTags: List[(Int, Int)]): List[Int] = {
    val indexAnnotation = indexTags.zipWithIndex.collect {
      case (range, index) if limits._1 - 1 <= range._1 && limits._2 > range._2 => index
    }
    indexAnnotation
  }

  private def getPhrase(indexAnnotation: List[Int], annotations: Seq[Annotation]): Seq[Annotation] = {
    val annotation = indexAnnotation.map(index => annotations.apply(index))
    annotation
  }

  private def getChunkPhrases(POSTagPattern: Regex, POSFormatSentence: String, annotations: Seq[Annotation]):
  Option[Array[Seq[Annotation]]] = {
    val rangeMatches = patternMatchIndexes(POSTagPattern, POSFormatSentence)
    if (rangeMatches.isEmpty){
      None
    }
    val indexLeftTags = patternMatchFirstIndex("<".r, POSFormatSentence)
    val indexRightTags = patternMatchFirstIndex(">".r, POSFormatSentence)
    val indexTags = indexLeftTags zip indexRightTags //merge two sequential collections
    val indexAnnotations = rangeMatches.map(range => getIndexAnnotation(range, indexTags))
    Some(indexAnnotations.map(indexAnnotation => getPhrase(indexAnnotation, annotations)).toArray)
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val sentences = annotations.filter(_.annotatorType == AnnotatorType.DOCUMENT)

    sentences.zipWithIndex.flatMap { case(sentence, sentenceIndex) =>

      val sentencePos = annotations.filter(pos =>
        pos.annotatorType == AnnotatorType.POS &&
          pos.begin >= sentence.begin &&
          pos.end <= sentence.end)

      val POSFormatSentence = sentencePos.map(annotation => "<"+annotation.result+">")
        .mkString(" ").replaceAll("\\s","")

      val chunkPhrases = POSTagPatterns.flatMap(POSTagPattern =>
        getChunkPhrases(POSTagPattern, POSFormatSentence, sentencePos)).flatten

      val chunkAnnotations = chunkPhrases.zipWithIndex.map{ case (phrase, idx) =>
        val result = sentence.result.substring(
          phrase.head.begin - sentence.begin,
          phrase.last.end - sentence.begin + 1
        )
        val start = phrase.head.begin
        val end = phrase.last.end
        Annotation(
          outputAnnotatorType,
          start,
          end,
          result,
          Map("sentence" -> sentenceIndex.toString, "chunk" -> idx.toString)
        )
      }

      chunkAnnotations
    }

  }

}

object Chunker extends DefaultParamsReadable[Chunker]