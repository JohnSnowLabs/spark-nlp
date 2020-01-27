package com.johnsnowlabs.nlp.annotators

import java.util.Locale

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.param.{BooleanParam, Param, ParamValidators, StringArrayParam}
import org.apache.spark.ml.util.Identifiable

class StopWordsCleaner(override val uid: String) extends AnnotatorModel[StopWordsCleaner] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val outputAnnotatorType: AnnotatorType = TOKEN

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("STOPWORDS_CLEANER"))

  val stopWords: StringArrayParam =
    new StringArrayParam(this, "stopWords", "the words to be filtered out. by default it's english stop words from Spark ML")

  def setStopWords(value: Array[String]): this.type = set(stopWords, value)
  def getStopWords: Array[String] = $(stopWords)

  val caseSensitive: BooleanParam = new BooleanParam(this, "caseSensitive",
    "whether to do a case-sensitive comparison over the stop words")

  def setCaseSensitive(value: Boolean): this.type = set(caseSensitive, value)
  def getCaseSensitive: Boolean = $(caseSensitive)

  val locale: Param[String] = new Param[String](this, "locale",
    "Locale of the input for case insensitive matching. Ignored when caseSensitive is true.",
    ParamValidators.inArray[String](Locale.getAvailableLocales.map(_.toString)))

  def setLocale(value: String): this.type = set(locale, value)
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