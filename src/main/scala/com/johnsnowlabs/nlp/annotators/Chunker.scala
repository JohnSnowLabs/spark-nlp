/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, HasSimpleAnnotate}
import org.apache.spark.ml.param.StringArrayParam
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.util.matching.Regex

/** This annotator matches a pattern of part-of-speech tags in order to return meaningful phrases
  * from document. Extracted part-of-speech tags are mapped onto the sentence, which can then be
  * parsed by regular expressions. The part-of-speech tags are wrapped by angle brackets `<>` to
  * be easily distinguishable in the text itself. This example sentence will result in the form:
  * {{{
  * "Peter Pipers employees are picking pecks of pickled peppers."
  * "<NNP><NNP><NNS><VBP><VBG><NNS><IN><JJ><NNS><.>"
  * }}}
  * To then extract these tags, `regexParsers` need to be set with e.g.:
  * {{{
  * val chunker = new Chunker()
  *   .setInputCols("sentence", "pos")
  *   .setOutputCol("chunk")
  *   .setRegexParsers(Array("<NNP>+", "<NNS>+"))
  * }}}
  * When defining the regular expressions, tags enclosed in angle brackets are treated as groups,
  * so here specifically `"<NNP>+"` means 1 or more nouns in succession. Additional patterns can
  * also be set with `addRegexParsers`.
  *
  * For more extended examples see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb Spark NLP Workshop]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ChunkerTestSpec.scala ChunkerTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.{Chunker, Tokenizer}
  * import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
  * import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentence = new SentenceDetector()
  *   .setInputCols("document")
  *   .setOutputCol("sentence")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols(Array("sentence"))
  *   .setOutputCol("token")
  *
  * val POSTag = PerceptronModel.pretrained()
  *   .setInputCols("document", "token")
  *   .setOutputCol("pos")
  *
  * val chunker = new Chunker()
  *   .setInputCols("sentence", "pos")
  *   .setOutputCol("chunk")
  *   .setRegexParsers(Array("<NNP>+", "<NNS>+"))
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(
  *     documentAssembler,
  *     sentence,
  *     tokenizer,
  *     POSTag,
  *     chunker
  *   ))
  *
  * val data = Seq("Peter Pipers employees are picking pecks of pickled peppers.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(chunk) as result").show(false)
  * +-------------------------------------------------------------+
  * |result                                                       |
  * +-------------------------------------------------------------+
  * |[chunk, 0, 11, Peter Pipers, [sentence -> 0, chunk -> 0], []]|
  * |[chunk, 13, 21, employees, [sentence -> 0, chunk -> 1], []]  |
  * |[chunk, 35, 39, pecks, [sentence -> 0, chunk -> 2], []]      |
  * |[chunk, 52, 58, peppers, [sentence -> 0, chunk -> 3], []]    |
  * +-------------------------------------------------------------+
  * }}}
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel PerceptronModel]] for
  *   Part-Of-Speech tagging
  * @param uid
  *   internal uid required to generate writable annotators
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
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
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class Chunker(override val uid: String)
    extends AnnotatorModel[Chunker]
    with HasSimpleAnnotate[Chunker] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** an array of grammar based chunk parsers
    *
    * @group param
    */
  val regexParsers =
    new StringArrayParam(this, "regexParsers", "an array of grammar based chunk parsers")

  /** Output annotator type : CHUNK
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = CHUNK

  /** Input annotator type : DOCUMENT, POS
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, POS)

  /** A list of regex patterns to match chunks, for example: Array(“‹DT›?‹JJ›*‹NN›”)
    *
    * @group setParam
    */
  def setRegexParsers(value: Array[String]): Chunker = set(regexParsers, value)

  /** adds a pattern to the current list of chunk patterns, for example: “‹DT›?‹JJ›*‹NN›”
    *
    * @group setParam
    */
  def addRegexParser(value: String): Chunker = {
    set(regexParsers, get(regexParsers).getOrElse(Array.empty[String]) :+ value)
  }

  /** A list of regex patterns to match chunks, for example: Array(“‹DT›?‹JJ›*‹NN›”)
    *
    * @group getParam
    */
  def getRegexParsers: Array[String] = $(regexParsers)

  def this() = this(Identifiable.randomUID("CHUNKER"))

  /** @group param */
  private lazy val replacements = Map("<" -> "(?:<", ">" -> ">)", "|" -> ">|<")
  private lazy val emptyString = ""

  /** @group param */
  private lazy val POSTagPatterns: Array[Regex] = {
    getRegexParsers.map(regexParser => replaceRegexParser(regexParser))
  }

  private def replaceRegexParser(regexParser: String): Regex = {
    replacements
      .foldLeft(regexParser)((accumulatedParser, keyValueReplace) =>
        accumulatedParser.replaceAllLiterally(keyValueReplace._1, keyValueReplace._2))
      .r
  }

  private def patternMatchIndexes(pattern: Regex, text: String): List[(Int, Int)] = {
    pattern.findAllMatchIn(text).map(index => (index.start, index.end)).toList
  }

  private def patternMatchFirstIndex(pattern: Regex, text: String): List[Int] =
    pattern.findAllMatchIn(text).map(_.start).toList

  private def getIndexAnnotation(limits: (Int, Int), indexTags: List[(Int, Int)]): List[Int] = {
    val indexAnnotation = indexTags.zipWithIndex.collect {
      case (range, index) if limits._1 - 1 <= range._1 && limits._2 > range._2 => index
    }
    indexAnnotation
  }

  private def getPhrase(
      indexAnnotation: List[Int],
      annotations: Seq[Annotation]): Seq[Annotation] = {
    val annotation = indexAnnotation.map(index => annotations.apply(index))
    annotation
  }

  private def getChunkPhrases(
      POSTagPattern: Regex,
      POSFormatSentence: String,
      annotations: Seq[Annotation]): Option[Array[Seq[Annotation]]] = {
    val rangeMatches = patternMatchIndexes(POSTagPattern, POSFormatSentence)
    if (rangeMatches.isEmpty) {
      None
    }
    val indexLeftTags = patternMatchFirstIndex("<".r, POSFormatSentence)
    val indexRightTags = patternMatchFirstIndex(">".r, POSFormatSentence)
    val indexTags = indexLeftTags zip indexRightTags // merge two sequential collections
    val indexAnnotations = rangeMatches.map(range => getIndexAnnotation(range, indexTags))
    Some(indexAnnotations.map(indexAnnotation => getPhrase(indexAnnotation, annotations)).toArray)
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val sentences = annotations.filter(_.annotatorType == AnnotatorType.DOCUMENT)

    sentences.zipWithIndex.flatMap { case (sentence, sentenceIndex) =>
      val sentencePos = annotations.filter(pos =>
        pos.annotatorType == AnnotatorType.POS &&
          pos.begin >= sentence.begin &&
          pos.end <= sentence.end)

      val POSFormatSentence = sentencePos
        .map(annotation => "<" + annotation.result + ">")
        .mkString(" ")
        .replaceAll("\\s", "")

      val chunkPhrases = POSTagPatterns
        .flatMap(POSTagPattern => getChunkPhrases(POSTagPattern, POSFormatSentence, sentencePos))
        .flatten

      val chunkAnnotations = chunkPhrases.zipWithIndex.map { case (phrase, idx) =>
        /** avoid exception if any document/sentence is dirty with bad indices */
        val result =
          try {
            sentence.result.substring(
              phrase.head.begin - sentence.begin,
              phrase.last.end - sentence.begin + 1)
          } catch {
            case _: Exception => emptyString
          }

        val start = phrase.head.begin
        val end = phrase.last.end
        Annotation(
          outputAnnotatorType,
          start,
          end,
          result,
          Map("sentence" -> sentenceIndex.toString, "chunk" -> idx.toString))
      }

      /** filter out any annotation with empty result */
      chunkAnnotations.filter(x => x.result.nonEmpty)
    }

  }

}

/** This is the companion object of [[Chunker]]. Please refer to that class for the documentation.
  */
object Chunker extends DefaultParamsReadable[Chunker]
