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

package com.johnsnowlabs.nlp.annotators.spell.norvig

import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{AnalysisException, Dataset}

/** Trains annotator, that retrieves tokens and makes corrections automatically if not found in an
  * English dictionary, based on the algorithm by Peter Norvig.
  *
  * The algorithm is based on a Bayesian approach to spell checking: Given the word we look in the
  * provided dictionary to choose the word with the highest probability to be the correct one.
  *
  * A dictionary of correct spellings must be provided with `setDictionary` either in the form of
  * a text file or directly as an
  * [[com.johnsnowlabs.nlp.util.io.ExternalResource ExternalResource]], where each word is parsed
  * by a regex pattern.
  *
  * Inspired by the spell checker by Peter Norvig:
  * [[https://norvig.com/spell-correct.html How to Write a Spelling Corrector]].
  *
  * For instantiated/pretrained models, see [[NorvigSweetingModel]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/vivekn-sentiment/VivekNarayanSentimentApproach.ipynb Spark NLP Workshop]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingTestSpec.scala NorvigSweetingTestSpec]].
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
  * import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
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
  * val spellChecker = new NorvigSweetingApproach()
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
  *   [[com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach SymmetricDeleteApproach]]
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
class NorvigSweetingApproach(override val uid: String)
    extends AnnotatorApproach[NorvigSweetingModel]
    with NorvigSweetingParams {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Spell checking algorithm inspired on Norvig model */
  override val description: String = "Spell checking algorithm inspired on Norvig model"

  /** External dictionary to be used, which needs `"tokenPattern"` (Default: `\S+`) for parsing
    * the resource.
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
    new ExternalResourceParam(this, "dictionary", "File with a list of correct words")

  setDefault(
    caseSensitive -> true,
    doubleVariants -> false,
    shortCircuit -> false,
    frequencyPriority -> true,
    wordSizeIgnore -> 3,
    dupsLimit -> 2,
    reductLimit -> 3,
    intersections -> 10,
    vowelSwapLimit -> 6)

  /** External dictionary already in the form of [[ExternalResource]], for which the Map member
    * `options` has an entry defined for `"tokenPattern"`.
    * ==Example==
    * {{{
    * val resource = ExternalResource(
    *   "src/test/resources/spell/words.txt",
    *   ReadAs.TEXT,
    *   Map("tokenPattern" -> "\\S+")
    * )
    * val spellChecker = new NorvigSweetingApproach()
    *   .setInputCols("token")
    *   .setOutputCol("spell")
    *   .setDictionary(resource)
    * }}}
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

  def this() = this(Identifiable.randomUID("SPELL"))

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): NorvigSweetingModel = {

    validateDataSet(dataset)
    val loadWords = ResourceHelper.getWordCount($(dictionary)).toMap
    val corpusWordCount: Map[String, Long] = {

      dataset
        .select(getInputCols.head)
        .as[Array[Annotation]]
        .flatMap(_.map(_.result))
        .groupBy("value")
        .count
        .as[(String, Long)]
        .collect
        .toMap
    }

    new NorvigSweetingModel()
      .setWordSizeIgnore($(wordSizeIgnore))
      .setDupsLimit($(dupsLimit))
      .setReductLimit($(reductLimit))
      .setIntersections($(intersections))
      .setVowelSwapLimit($(vowelSwapLimit))
      .setWordCount(loadWords ++ corpusWordCount)
      .setDoubleVariants($(doubleVariants))
      .setCaseSensitive($(caseSensitive))
      .setShortCircuit($(shortCircuit))
      .setFrequencyPriority($(frequencyPriority))
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

/** This is the companion object of [[NorvigSweetingApproach]]. Please refer to that class for the
  * documentation.
  */
object NorvigSweetingApproach extends DefaultParamsReadable[NorvigSweetingApproach]
