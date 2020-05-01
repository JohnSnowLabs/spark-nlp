package com.johnsnowlabs.nlp.annotators

import java.util.Locale

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.param.{BooleanParam, Param, ParamValidators, StringArrayParam}
import org.apache.spark.ml.util.Identifiable

/** This annotator excludes from a sequence of strings (e.g. the output of a Tokenizer, Normalizer, Lemmatizer, and Stemmer) and drops all the stop words from the input sequences.
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/StopWordsCleanerTestSpec.scala]] for example of how to use this API.
  * */
class StopWordsCleaner(override val uid: String) extends AnnotatorModel[StopWordsCleaner] {

  import com.johnsnowlabs.nlp.AnnotatorType._


  /** Output annotator type: TOKEN */
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** Input annotator type: TOKEN */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("STOPWORDS_CLEANER"))

  /** the words to be filtered out. by default it's english stop words from Spark ML */
  val stopWords: StringArrayParam = new StringArrayParam(this, "stopWords", "the words to be filtered out. by default it's english stop words from Spark ML")

  /** The words to be filtered out  */
  def setStopWords(value: Array[String]): this.type = set(stopWords, value)

  /** The words to be filtered out  */
  def getStopWords: Array[String] = $(stopWords)

  val caseSensitive: BooleanParam = new BooleanParam(this, "caseSensitive",
    "whether to do a case-sensitive comparison over the stop words")

  /** Whether to do a case sensitive comparison over the stop words.  */
  def setCaseSensitive(value: Boolean): this.type = set(caseSensitive, value)

  /** Whether to do a case sensitive comparison over the stop words.  */
  def getCaseSensitive: Boolean = $(caseSensitive)

  /** Locale of the input for case insensitive matching. Ignored when caseSensitive is true. */
  val locale: Param[String] = new Param[String](this, "locale", "Locale of the input for case insensitive matching. Ignored when caseSensitive is true.",
    ParamValidators.inArray[String](Locale.getAvailableLocales.map(_.toString)))

  /** Locale of the input for case insensitive matching. Ignored when caseSensitive is true */
  def setLocale(value: String): this.type = set(locale, value)

  /** Locale of the input for case insensitive matching. Ignored when caseSensitive is true */
  def getLocale: String = $(locale)

  /**
    * Returns system default locale, or `Locale.US` if the default locale is not in available locales
    * in JVM.
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
    locale -> getDefaultOrUS.toString
  )

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
        tokenAnnotation.metadata
      )
    }
  }

}

object StopWordsCleaner extends ParamsAndFeaturesReadable[StopWordsCleaner]