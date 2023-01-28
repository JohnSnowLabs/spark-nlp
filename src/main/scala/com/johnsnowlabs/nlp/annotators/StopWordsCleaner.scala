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

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.param.{BooleanParam, Param, ParamValidators, StringArrayParam}
import org.apache.spark.ml.util.Identifiable

import java.util.Locale

/** This annotator takes a sequence of strings (e.g. the output of a [[Tokenizer]],
  * [[Normalizer]], [[Lemmatizer]], and [[Stemmer]]) and drops all the stop words from the input
  * sequences.
  *
  * By default, it uses stop words from MLlibs
  * [[https://spark.apache.org/docs/latest/ml-features#stopwordsremover StopWordsRemover]]. Stop
  * words can also be defined by explicitly setting them with `setStopWords(value: Array[String])`
  * or loaded from pretrained models using `pretrained` of its companion object.
  * {{{
  * val stopWords = StopWordsCleaner.pretrained()
  *   .setInputCols("token")
  *   .setOutputCol("cleanTokens")
  *   .setCaseSensitive(false)
  * // will load the default pretrained model `"stopwords_en"`.
  * }}}
  * For available pretrained models please see the
  * [[https://nlp.johnsnowlabs.com/models?task=Stop+Words+Removal Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/annotation/text/english/stop-words/StopWordsCleaner.ipynb Examples]]
  * and
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/StopWordsCleanerTestSpec.scala StopWordsCleanerTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.Tokenizer
  * import com.johnsnowlabs.nlp.annotator.SentenceDetector
  * import com.johnsnowlabs.nlp.annotators.StopWordsCleaner
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentenceDetector = new SentenceDetector()
  *   .setInputCols(Array("document"))
  *   .setOutputCol("sentence")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols(Array("sentence"))
  *   .setOutputCol("token")
  *
  * val stopWords = new StopWordsCleaner()
  *   .setInputCols("token")
  *   .setOutputCol("cleanTokens")
  *   .setCaseSensitive(false)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *     documentAssembler,
  *     sentenceDetector,
  *     tokenizer,
  *     stopWords
  *   ))
  *
  * val data = Seq(
  *   "This is my first sentence. This is my second.",
  *   "This is my third sentence. This is my forth."
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("cleanTokens.result").show(false)
  * +-------------------------------+
  * |result                         |
  * +-------------------------------+
  * |[first, sentence, ., second, .]|
  * |[third, sentence, ., forth, .] |
  * +-------------------------------+
  * }}}
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
class StopWordsCleaner(override val uid: String)
    extends AnnotatorModel[StopWordsCleaner]
    with HasSimpleAnnotate[StopWordsCleaner] {

  /** Output annotator type: TOKEN
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** Input annotator type: TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("STOPWORDS_CLEANER"))

  /** The words to be filtered out (Default: Stop words from MLlib)
    *
    * @group param
    */
  val stopWords: StringArrayParam = new StringArrayParam(
    this,
    "stopWords",
    "The words to be filtered out. by default it's english stop words from Spark ML")

  /** The words to be filtered out
    *
    * @group setParam
    */
  def setStopWords(value: Array[String]): this.type = set(stopWords, value)

  /** The words to be filtered out
    *
    * @group getParam
    */
  def getStopWords: Array[String] = $(stopWords)

  /** Whether to do a case-sensitive comparison over the stop words (Default: `false`)
    *
    * @group param
    */
  val caseSensitive: BooleanParam = new BooleanParam(
    this,
    "caseSensitive",
    "Whether to do a case-sensitive comparison over the stop words")

  /** Whether to do a case-sensitive comparison over the stop words (Default: `false`)
    *
    * @group setParam
    */
  def setCaseSensitive(value: Boolean): this.type = set(caseSensitive, value)

  /** Whether to do a case-sensitive comparison over the stop words (Default: `false`)
    *
    * @group getParam
    */
  def getCaseSensitive: Boolean = $(caseSensitive)

  /** Locale of the input for case insensitive matching (Default: system default locale, or
    * `Locale.US` if the default locale is not in available locales). Ignored when caseSensitive
    * is true.
    * @group param
    */
  val locale: Param[String] = new Param[String](
    this,
    "locale",
    "Locale of the input for case insensitive matching. Ignored when caseSensitive is true.",
    ParamValidators.inArray[String](Locale.getAvailableLocales.map(_.toString)))

  /** Locale of the input for case insensitive matching (Default: system default locale, or
    * `Locale.US` if the default locale is not in available locales). Ignored when caseSensitive
    * is true
    * @group setParam
    */
  def setLocale(value: String): this.type = set(locale, value)

  /** Locale of the input for case insensitive matching (Default: system default locale, or
    * `Locale.US` if the default locale is not in available locales). Ignored when caseSensitive
    * is true
    * @group getParam
    */
  def getLocale: String = $(locale)

  /** Returns system default locale, or `Locale.US` if the default locale is not in available
    * locales in JVM.
    *
    * @group param
    */
  private val getDefaultOrUS: Locale = {
    if (Locale.getAvailableLocales.contains(Locale.getDefault)) {
      Locale.getDefault
    } else {
      logWarning(s"Default locale set was [${Locale.getDefault.toString}]; however, it was " +
        "not found in available locales in JVM, falling back to en_US locale. Set param `locale` " +
        "in order to respect another locale.")
      Locale.US
    }
  }

  setDefault(
    inputCols -> Array(TOKEN),
    outputCol -> "cleanedToken",
    stopWords -> StopWordsRemover.loadDefaultStopWords("english"),
    caseSensitive -> false,
    locale -> getDefaultOrUS.toString)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val annotationsWithoutStopWords = if ($(caseSensitive)) {
      val stopWordsSet = $(stopWords).toSet
      annotations.filter(s => !stopWordsSet.contains(s.result))
    } else {
      val lc = new Locale($(locale))
      // scalastyle:off caselocale
      val toLower = (s: String) => if (s != null) s.toLowerCase(lc) else s
      val lowerStopWords = $(stopWords).map(toLower(_)).toSet
      annotations.filter(s => !lowerStopWords.contains(toLower(s.result)))
    }

    annotationsWithoutStopWords.map { tokenAnnotation =>
      Annotation(
        outputAnnotatorType,
        tokenAnnotation.begin,
        tokenAnnotation.end,
        tokenAnnotation.result,
        tokenAnnotation.metadata)
    }
  }

}

trait ReadablePretrainedStopWordsCleanerModel
    extends ParamsAndFeaturesReadable[StopWordsCleaner]
    with HasPretrained[StopWordsCleaner] {
  override val defaultModelName: Some[String] = Some("stopwords_en")

  /** Java compliant-overrides */
  override def pretrained(): StopWordsCleaner = super.pretrained()
  override def pretrained(name: String): StopWordsCleaner = super.pretrained(name)
  override def pretrained(name: String, lang: String): StopWordsCleaner =
    super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): StopWordsCleaner =
    super.pretrained(name, lang, remoteLoc)
}

object StopWordsCleaner
    extends ParamsAndFeaturesReadable[StopWordsCleaner]
    with ReadablePretrainedStopWordsCleanerModel
