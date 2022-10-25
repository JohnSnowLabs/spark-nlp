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

package com.johnsnowlabs.nlp.annotators.spell.symmetric

import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{AnalysisException, Dataset}

import scala.collection.mutable.ListBuffer

/** Trains a Symmetric Delete spelling correction algorithm. Retrieves tokens and utilizes
  * distance metrics to compute possible derived words.
  *
  * The Symmetric Delete spelling correction algorithm reduces the complexity of edit candidate
  * generation and dictionary lookup for a given Damerau-Levenshtein distance. It is six orders of
  * magnitude faster (than the standard approach with deletes + transposes + replaces + inserts)
  * and language independent. A dictionary of correct spellings must be provided with
  * `setDictionary` either in the form of a text file or directly as an
  * [[com.johnsnowlabs.nlp.util.io.ExternalResource ExternalResource]], where each word is parsed
  * by a regex pattern.
  *
  * Inspired by [[https://github.com/wolfgarbe/SymSpell SymSpell]].
  *
  * For instantiated/pretrained models, see [[SymmetricDeleteModel]].
  *
  * See
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteModelTestSpec.scala SymmetricDeleteModelTestSpec]]
  * for further reference.
  *
  * ==Example==
  * In this example, the dictionary `"words.txt"` has the form of
  * {{{
  * ...
  * gummy
  * gummic
  * gummier
  * gummiest
  * gummiferous
  * ...
  * }}}
  * This dictionary is then set to be the basis of the spell checker.
  * {{{
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val spellChecker = new SymmetricDeleteApproach()
  *   .setInputCols("token")
  *   .setOutputCol("spell")
  *   .setDictionary("src/test/resources/spell/words.txt")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   spellChecker
  * ))
  *
  * val pipelineModel = pipeline.fit(trainingData)
  * }}}
  *
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach NorvigSweetingApproach]]
  *   for an alternative approach to spell checking
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerApproach ContextSpellCheckerApproach]]
  *   for a DL based approach
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
class SymmetricDeleteApproach(override val uid: String)
    extends AnnotatorApproach[SymmetricDeleteModel]
    with SymmetricDeleteParams {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Spell checking algorithm inspired on Symmetric Delete algorithm */
  override val description: String =
    "Spell checking algorithm inspired on Symmetric Delete algorithm"

  /** Optional dictionary of properly written words. If provided, significantly boosts spell
    * checking performance.
    *
    * Needs `"tokenPattern"` (Default: `\S+`) for parsing the resource.
    * ==Example==
    * {{{
    * ...
    * gummy
    * gummic
    * gummier
    * gummiest
    * gummiferous
    * ...
    * }}}
    *
    * @group param
    */
  val dictionary =
    new ExternalResourceParam(this, "dictionary", "file with a list of correct words")

  setDefault(frequencyThreshold -> 0, deletesThreshold -> 0, maxEditDistance -> 3, dupsLimit -> 2)

  /** External dictionary already in the form of [[ExternalResource]], for which the Map member
    * `options` has an entry defined for `"tokenPattern"`.
    * ==Example==
    * {{{
    * val resource = ExternalResource(
    *   "src/test/resources/spell/words.txt",
    *   ReadAs.TEXT,
    *   Map("tokenPattern" -> "\\S+")
    * )
    * val spellChecker = new SymmetricDeleteApproach()
    *   .setInputCols("token")
    *   .setOutputCol("spell")
    *   .setDictionary(resource)
    * }}}
    *
    * @group setParam
    */
  def setDictionary(value: ExternalResource): this.type = {
    require(
      value.options.contains("tokenPattern"),
      "dictionary needs 'tokenPattern' regex in dictionary for separating words")
    set(dictionary, value)
  }

  /** Path to file with properly spelled words, `tokenPattern` is the regex pattern to identify
    * them in text, readAs can be `ReadAs.TEXT` or `ReadAs.SPARK`, with options passed to Spark
    * reader if the latter is set. Dictionary needs `tokenPattern` regex for separating words.
    *
    * @group setParam
    */
  def setDictionary(
      path: String,
      tokenPattern: String = "\\S+",
      readAs: ReadAs.Format = ReadAs.TEXT,
      options: Map[String, String] = Map("format" -> "text")): this.type =
    set(
      dictionary,
      ExternalResource(path, readAs, options ++ Map("tokenPattern" -> tokenPattern)))

  /** Output annotator type : TOKEN
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** Input annotator type : TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() =
    this(
      Identifiable.randomUID("SYMSPELL")
    ) // constructor required for the annotator to work in python

  /** Given a word, derive strings with up to maxEditDistance characters deleted */
  def getDeletes(word: String, med: Int): List[String] = {

    var deletes = new ListBuffer[String]()
    var queueList = List(word)
    val x = 1 to med
    x.foreach(_ => {
      var tempQueue = new ListBuffer[String]()
      queueList.foreach(w => {
        if (w.length > 1) {
          val y = 0 until w.length
          y.foreach(c => { // character index
            // result of word minus c
            val wordMinus = w.substring(0, c).concat(w.substring(c + 1, w.length))
            if (!deletes.contains(wordMinus)) {
              deletes += wordMinus
            }
            if (!tempQueue.contains(wordMinus)) {
              tempQueue += wordMinus
            }
          }) // End y.foreach
          queueList = tempQueue.toList
        }
      }) // End queueList.foreach
    }) // End x.foreach

    deletes.toList
  }

  /** Computes derived words from a frequency of words */
  def derivedWordDistances(
      wordFrequencies: List[(String, Long)],
      maxEditDistance: Int): Map[String, (List[String], Long)] = {

    val derivedWords = scala.collection.mutable.Map(wordFrequencies.map { a =>
      (a._1, (ListBuffer.empty[String], a._2))
    }: _*)

    wordFrequencies.foreach { case (word, _) =>
      val deletes = getDeletes(word, maxEditDistance)

      deletes.foreach(deleteItem => {
        if (derivedWords.contains(deleteItem)) {
          // add (correct) word to delete's suggested correction list
          derivedWords(deleteItem)._1 += word
        } else {
          // note frequency of word in corpus is not incremented
          derivedWords(deleteItem) = (ListBuffer(word), 0L)
        }
      }) // End deletes.foreach
    }
    derivedWords
      .filterKeys(a => derivedWords(a)._1.length >= $(deletesThreshold))
      .mapValues(derivedWords => (derivedWords._1.toList, derivedWords._2))
      .toMap
  }

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): SymmetricDeleteModel = {

    require(!dataset.rdd.isEmpty(), "Dataset for training is empty")

    validateDataSet(dataset)

    val possibleDict = get(dictionary).map(d => ResourceHelper.getWordCount(d))

    val trainDataSet =
      dataset
        .select(getInputCols.head)
        .as[Array[Annotation]]
        .flatMap(_.map(_.result))

    val wordFrequencies =
      trainDataSet
        .groupBy("value")
        .count()
        .filter(s"count(value) >= ${$(frequencyThreshold)}")
        .as[(String, Long)]
        .collect
        .toList

    val derivedWords =
      derivedWordDistances(wordFrequencies, $(maxEditDistance))

    val longestWordLength =
      trainDataSet.agg(max(length(col("value")))).head().getInt(0)

    val model =
      new SymmetricDeleteModel()
        .setDerivedWords(derivedWords)
        .setLongestWordLength(longestWordLength)

    if (possibleDict.isDefined) {
      val min = wordFrequencies.minBy(_._2)._2
      val max = wordFrequencies.maxBy(_._2)._2
      model.setMinFrequency(min)
      model.setMaxFrequency(max)
      model.setDictionary(possibleDict.get.toMap)
    }

    model
  }

  private def validateDataSet(dataset: Dataset[_]): Unit = {
    try {
      dataset.select(getInputCols.head).as[Array[Annotation]]
    } catch {
      case exception: AnalysisException =>
        if (exception.getMessage == "need an array field but got string;") {
          throw new IllegalArgumentException(
            "Train dataset must have an array annotation type column")
        }
        throw exception
    }
  }

}
// This objects reads the class' properties, it enables reading the model after it is stored

/** This is the companion object of [[SymmetricDeleteApproach]]. Please refer to that class for
  * the documentation.
  */
object SymmetricDeleteApproach extends DefaultParamsReadable[SymmetricDeleteApproach]
