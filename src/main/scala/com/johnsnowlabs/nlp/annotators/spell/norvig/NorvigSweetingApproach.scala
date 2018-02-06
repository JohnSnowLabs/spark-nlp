package com.johnsnowlabs.nlp.annotators.spell.norvig

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ResourceHelper, ReadAs}
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

  setDefault(dictionary, ExternalResource(
    "/spell/words.txt",
    ReadAs.LINE_BY_LINE,
    options=Map("tokenPattern" -> "[a-zA-Z]+"))
  )

  setDefault(caseSensitive, false)
  setDefault(doubleVariants, false)
  setDefault(shortCircuit, false)

  def setCorpus(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "spell checker corpus needs 'tokenPattern' regex for tagging words. e.g. [a-zA-Z]+")
    set(corpus, value)
  }

  def setDictionary(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "dictionary needs 'tokenPattern' regex in dictionary for separating words")
    set(dictionary, value)
  }

  def setSlangDictionary(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "slang dictionary is a delimited text. needs 'delimiter' in options")
    set(slangDictionary, value)
  }

  override val annotatorType: AnnotatorType = SPELL

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("SPELL"))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NorvigSweetingModel = {
    val loadWords = ResourceHelper.wordCount($(dictionary))
    val corpusWordCount =
      if (get(corpus).isDefined) {
        ResourceHelper.wordCount($(corpus))
      } else {
      Map.empty[String, Int]
      }
    val loadSlangs = if (get(slangDictionary).isDefined)
      ResourceHelper.parseKeyValueText($(slangDictionary))
    else
      Map.empty[String, String]
    new NorvigSweetingModel()
      .setWordCount(loadWords.toMap ++ corpusWordCount)
      .setCustomDict(loadSlangs)
      .setDoubleVariants($(doubleVariants))
      .setCaseSensitive($(caseSensitive))
      .setShortCircuit($(shortCircuit))
  }

}
object NorvigSweetingApproach extends DefaultParamsReadable[NorvigSweetingApproach]