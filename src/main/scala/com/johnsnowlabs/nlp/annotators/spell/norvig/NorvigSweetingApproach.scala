package com.johnsnowlabs.nlp.annotators.spell.norvig

import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class NorvigSweetingApproach(override val uid: String)
  extends AnnotatorApproach[NorvigSweetingModel]
    with NorvigSweetingParams {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val description: String = "Spell checking algorithm inspired on Norvig model"

  val corpus = new ExternalResourceParam(this, "corpus", "folder or file with text that teaches about the language")
  val dictionary = new ExternalResourceParam(this, "dictionary", "file with a list of correct words")
  val slangDictionary = new ExternalResourceParam(this, "slangDictionary", "delimited file with list of custom words to be manually corrected")

  setDefault(caseSensitive, false)
  setDefault(doubleVariants, false)
  setDefault(shortCircuit, false)

  def setCorpus(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "spell checker corpus needs 'tokenPattern' regex for tagging words. e.g. [a-zA-Z]+")
    set(corpus, value)
  }

  def setCorpus(path: String,
                tokenPattern: String,
                readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                options: Map[String, String] = Map("format" -> "text")): this.type =
    set(corpus, ExternalResource(path, readAs, options ++ Map("tokenPattern" -> tokenPattern)))

  def setDictionary(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "dictionary needs 'tokenPattern' regex in dictionary for separating words")
    set(dictionary, value)
  }

  def setDictionary(path: String,
                    tokenPattern: String = "\\S+",
                    readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                    options: Map[String, String] = Map("format" -> "text")): this.type =
    set(dictionary, ExternalResource(path, readAs, options ++ Map("tokenPattern" -> tokenPattern)))

  def setSlangDictionary(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "slang dictionary is a delimited text. needs 'delimiter' in options")
    set(slangDictionary, value)
  }

  def setSlangDictionary(path: String,
                         delimiter: String,
                         readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                         options: Map[String, String] = Map("format" -> "text")): this.type =
    set(slangDictionary, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))

  override val annotatorType: AnnotatorType = SPELL

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("SPELL"))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NorvigSweetingModel = {
    val loadWords = ResourceHelper.wordCount($(dictionary)).toMap
    val corpusWordCount: Map[String, Long] =
      if (get(corpus).isDefined) {
        ResourceHelper.wordCount($(corpus), p = recursivePipeline).toMap
      } else {
        import ResourceHelper.spark.implicits._
        dataset.show()
        dataset.select($(inputCols).head).show
        dataset.select($(inputCols).head).as[Array[Annotation]]
          .flatMap(_.map(_.result))
          .groupBy("value").count
          .as[(String, Long)]
          .collect.toMap
      }
    val loadSlangs = if (get(slangDictionary).isDefined)
      ResourceHelper.parseKeyValueText($(slangDictionary))
    else
      Map.empty[String, String]
    new NorvigSweetingModel()
      .setWordCount(loadWords ++ corpusWordCount)
      .setCustomDict(loadSlangs)
      .setDoubleVariants($(doubleVariants))
      .setCaseSensitive($(caseSensitive))
      .setShortCircuit($(shortCircuit))
  }

}
object NorvigSweetingApproach extends DefaultParamsReadable[NorvigSweetingApproach]