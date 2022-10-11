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

package com.johnsnowlabs.nlp.annotators.ws

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{
  AveragedPerceptron,
  PerceptronPredictionUtils
}
import com.johnsnowlabs.nlp.annotators.ws.TagsType.{LEFT_BOUNDARY, MIDDLE, RIGHT_BOUNDARY}
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.RegexTokenizer
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.Identifiable

/** WordSegmenter which tokenizes non-english or non-whitespace separated texts.
  *
  * Many languages are not whitespace separated and their sentences are a concatenation of many
  * symbols, like Korean, Japanese or Chinese. Without understanding the language, splitting the
  * words into their corresponding tokens is impossible. The WordSegmenter is trained to
  * understand these languages and plit them into semantically correct parts.
  *
  * This is the instantiated model of the [[WordSegmenterApproach]]. For training your own model,
  * please see the documentation of that class.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val wordSegmenter = WordSegmenterModel.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("words_segmented")
  * }}}
  * The default model is `"wordseg_pku"`, default language is `"zh"`, if no values are provided.
  * For available pretrained models please see the
  * [[https://nlp.johnsnowlabs.com/models?task=Word+Segmentation Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/chinese/word_segmentation/words_segmenter_demo.ipynb Spark NLP Workshop]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/WordSegmenterTest.scala WordSegmenterTest]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.WordSegmenterModel
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val wordSegmenter = WordSegmenterModel.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   wordSegmenter
  * ))
  *
  * val data = Seq("然而，這樣的處理也衍生了一些問題。").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("token.result").show(false)
  * +--------------------------------------------------------+
  * |result                                                  |
  * +--------------------------------------------------------+
  * |[然而, ，, 這樣, 的, 處理, 也, 衍生, 了, 一些, 問題, 。    ]|
  * +--------------------------------------------------------+
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
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class WordSegmenterModel(override val uid: String)
    extends AnnotatorModel[WordSegmenterModel]
    with HasSimpleAnnotate[WordSegmenterModel]
    with PerceptronPredictionUtils {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("WORD_SEGMENTER"))

  /** POS model
    *
    * @group param
    */
  val model: StructFeature[AveragedPerceptron] =
    new StructFeature[AveragedPerceptron](this, "POS Model")

  val enableRegexTokenizer: BooleanParam = new BooleanParam(
    this,
    "enableRegexTokenizer",
    "Whether to use RegexTokenizer before segmentation. Useful for multilingual text")

  /** Indicates whether to convert all characters to lowercase before tokenizing (Default:
    * `false`).
    *
    * @group param
    */
  val toLowercase: BooleanParam = new BooleanParam(
    this,
    "toLowercase",
    "Indicates whether to convert all characters to lowercase before tokenizing.\n")

  /** Regex pattern used to match delimiters (Default: `"\\s+"`)
    *
    * @group param
    */
  val pattern: Param[String] = new Param(this, "pattern", "regex pattern used for tokenizing")

  /** @group getParam */
  def getModel: AveragedPerceptron = $$(model)

  /** @group setParam */
  def setModel(targetModel: AveragedPerceptron): this.type = set(model, targetModel)

  /** @group setParam */
  def setEnableRegexTokenizer(value: Boolean): this.type = set(enableRegexTokenizer, value)

  /** @group setParam */
  def setToLowercase(value: Boolean): this.type = set(toLowercase, value)

  /** @group setParam */
  def setPattern(value: String): this.type = set(pattern, value)

  setDefault(enableRegexTokenizer -> false, toLowercase -> false, pattern -> "\\s+")

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param annotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    if ($(enableRegexTokenizer)) {
      return segmentWithRegexAnnotator(annotations)
    }

    val sentences = SentenceSplit.unpack(annotations)
    val tokens = getTokenAnnotations(sentences)
    val tokenizedSentences = TokenizedWithSentence.unpack(annotations ++ tokens)
    val tagged = tag($$(model), tokenizedSentences.toArray)
    buildWordSegments(tagged)
  }

  private def segmentWithRegexAnnotator(annotatedSentences: Seq[Annotation]): Seq[Annotation] = {

    val outputCol = Identifiable.randomUID("regex_token")

    val regexTokenizer = new RegexTokenizer()
      .setInputCols(getInputCols)
      .setOutputCol(outputCol)
      .setToLowercase($(toLowercase))
      .setPattern($(pattern))

    val annotatedTokens = regexTokenizer.annotate(annotatedSentences)

    val segmentedResult = annotatedTokens.flatMap { annotatedToken =>
      val codePoint = annotatedToken.result.codePointAt(0)
      val unicodeScript = Character.UnicodeScript.of(codePoint)
      if (unicodeScript == Character.UnicodeScript.LATIN) {
        Seq(annotatedToken)
      } else {
        val sentenceIndex = annotatedToken.metadata("sentence")

        val annotatedSentence = Annotation(
          DOCUMENT,
          annotatedToken.begin,
          annotatedToken.end,
          annotatedToken.result,
          Map("sentence" -> sentenceIndex))
        val sentence = Sentence(
          annotatedToken.result,
          annotatedToken.begin,
          annotatedToken.end,
          sentenceIndex.toInt)
        val annotatedTokens = getTokenAnnotations(Seq(sentence))

        val tokenizedSentences =
          TokenizedWithSentence.unpack(annotatedTokens ++ Seq(annotatedSentence))
        val tagged = tag($$(model), tokenizedSentences.toArray)
        buildWordSegments(tagged)
      }
    }

    segmentedResult
  }

  private def getTokenAnnotations(annotation: Seq[Sentence]): Seq[Annotation] = {
    val tokens = annotation.flatMap { sentence =>
      val chars = sentence.content.split("")
      chars.zipWithIndex
        .map { case (char, index) =>
          val tokenIndex = index + sentence.start
          Annotation(
            TOKEN,
            tokenIndex,
            tokenIndex,
            char,
            Map("sentence" -> sentence.index.toString))
        }
        .filter(annotation => annotation.result != " ")
    }
    tokens
  }

  def buildWordSegments(taggedSentences: Array[TaggedSentence]): Seq[Annotation] = {
    taggedSentences.zipWithIndex.flatMap { case (taggedSentence, index) =>
      val tagsSentence = taggedSentence.tags.mkString("")
      val wordIndexesByMatchedGroups = getWordIndexesByMatchedGroups(tagsSentence)
      if (wordIndexesByMatchedGroups.isEmpty) {
        taggedSentence.indexedTaggedWords.map(indexedTaggedWord =>
          Annotation(
            TOKEN,
            indexedTaggedWord.begin,
            indexedTaggedWord.end,
            indexedTaggedWord.word,
            Map("sentence" -> index.toString)))
      } else {
        annotateSegmentWords(wordIndexesByMatchedGroups, taggedSentence, index)
      }
    }
  }

  private def getWordIndexesByMatchedGroups(tagsSentence: String): List[List[RegexTagsInfo]] = {
    val regexPattern = s"($LEFT_BOUNDARY)($MIDDLE*)*($RIGHT_BOUNDARY)".r
    regexPattern
      .findAllMatchIn(tagsSentence)
      .map(matchedResult => {
        val groups = (1 to matchedResult.groupCount).toList
        groups
          .map(g =>
            RegexTagsInfo(
              matchedResult.group(g),
              matchedResult.start(g),
              matchedResult.end(g),
              (matchedResult.end(g) / 2) - 1))
          .filter(regexTagsInfo => regexTagsInfo.estimatedIndex != -1)
      })
      .toList
  }

  private def annotateSegmentWords(
      wordIndexesByMatchedGroups: List[List[RegexTagsInfo]],
      taggedSentence: TaggedSentence,
      sentenceIndex: Int): Seq[Annotation] = {

    val singleTaggedWords =
      getSingleIndexedTaggedWords(wordIndexesByMatchedGroups, taggedSentence)
    val multipleTaggedWords = getMultipleTaggedWords(wordIndexesByMatchedGroups, taggedSentence)
    val segmentedTaggedWords = (singleTaggedWords ++ multipleTaggedWords)
      .sortWith(
        _.metadata.getOrElse("index", "-1").toInt < _.metadata.getOrElse("index", "-1").toInt)
    segmentedTaggedWords.map(segmentedTaggedWord =>
      Annotation(
        TOKEN,
        segmentedTaggedWord.begin,
        segmentedTaggedWord.end,
        segmentedTaggedWord.word,
        Map("sentence" -> sentenceIndex.toString)))
  }

  private def getSingleIndexedTaggedWords(
      wordIndexesByMatchedGroups: List[List[RegexTagsInfo]],
      taggedSentence: TaggedSentence): List[IndexedTaggedWord] = {
    val flattenWordIndexes = wordIndexesByMatchedGroups.flatMap(wordIndexGroup =>
      wordIndexGroup.map(wi => wi.estimatedIndex))
    val unmatchedTaggedWordsCandidates = taggedSentence.indexedTaggedWords.zipWithIndex
      .filter { case (_, index) =>
        !flattenWordIndexes.contains(index)
      }
      .map(_._1)
    val unmatchedTaggedWords =
      unmatchedTaggedWordsCandidates.filter(unmatchedTaggedWordCandidate =>
        !isMatchedWord(unmatchedTaggedWordCandidate, wordIndexesByMatchedGroups))
    unmatchedTaggedWords.toList
  }

  private def isMatchedWord(
      indexedTaggedWord: IndexedTaggedWord,
      regexTagsInfoList: List[List[RegexTagsInfo]]): Boolean = {
    val index = indexedTaggedWord.metadata.getOrElse("index", "-1").toInt

    val result = regexTagsInfoList.flatMap(regexTagsInfo => {
      val leftBoundaryIndex = regexTagsInfo.head.estimatedIndex
      val rightBoundaryIndex = regexTagsInfo.last.estimatedIndex
      val isInRange = if (index > leftBoundaryIndex && index < rightBoundaryIndex) true else false
      val verifyMatches = regexTagsInfo.map(rti => {
        if (indexedTaggedWord.tag != MIDDLE || !isInRange) "unmatched"
        else {
          if (rti.tagsMatch.contains(MIDDLE) && rti.tagsMatch.length > 2) "matched"
          else "unmatched"
        }
      })
      verifyMatches
    })
    result.contains("matched")
  }

  private def getMultipleTaggedWords(
      wordIndexesByMatchedGroups: List[List[RegexTagsInfo]],
      taggedSentence: TaggedSentence): List[IndexedTaggedWord] = {
    wordIndexesByMatchedGroups.flatMap { wordIndexesGroup =>
      val wordIndexes = wordIndexesGroup.map(wi => wi.estimatedIndex)
      val taggedWords = taggedSentence.indexedTaggedWords.zipWithIndex
        .filter { case (indexedTaggedWord, index) =>
          wordIndexes.contains(index) || isMatchedWord(indexedTaggedWord, List(wordIndexesGroup))
        }
        .map(_._1)
      if (taggedWords.nonEmpty) Some(taggedWords.reduceLeft(processTags)) else None
    }
  }

  private val processTags = (current: IndexedTaggedWord, next: IndexedTaggedWord) => {
    val wordSegment = current.word + next.word
    val tagSegment = current.tag + next.tag
    val begin = if (current.begin <= next.begin) current.begin else next.begin
    val end = begin + wordSegment.length - 1
    val currentIndexValue = current.metadata.getOrElse("index", "-1")
    val nextIndexValue = current.metadata.getOrElse("index", "-1")
    val index =
      if (currentIndexValue.toInt <= nextIndexValue.toInt) currentIndexValue else nextIndexValue
    IndexedTaggedWord(wordSegment, tagSegment, begin, end, None, Map("index" -> index))
  }

  /** Output Annotator Types: TOKEN
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** Input Annotator Types: DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)
}

private case class RegexTagsInfo(tagsMatch: String, start: Int, end: Int, estimatedIndex: Int)

trait ReadablePretrainedWordSegmenter
    extends ParamsAndFeaturesReadable[WordSegmenterModel]
    with HasPretrained[WordSegmenterModel] {
  override val defaultModelName: Some[String] = Some("wordseg_pku")
  override val defaultLang: String = "zh"

  /** Java compliant-overrides */
  override def pretrained(): WordSegmenterModel = super.pretrained()

  override def pretrained(name: String): WordSegmenterModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): WordSegmenterModel =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): WordSegmenterModel =
    super.pretrained(name, lang, remoteLoc)
}

/** This is the companion object of [[WordSegmenterModel]]. Please refer to that class for the
  * documentation.
  */
object WordSegmenterModel extends ReadablePretrainedWordSegmenter
