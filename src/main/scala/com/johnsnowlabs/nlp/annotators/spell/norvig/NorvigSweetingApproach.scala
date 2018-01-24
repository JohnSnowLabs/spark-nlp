package com.johnsnowlabs.nlp.annotators.spell.norvig

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class NorvigSweetingApproach(override val uid: String)
  extends AnnotatorApproach[NorvigSweetingModel]
    with NorvigSweetingParams {

  import com.johnsnowlabs.nlp.AnnotatorType._
  import com.johnsnowlabs.nlp.util.io.ResourceFormat._

  override val description: String = "Spell checking algorithm inspired on Norvig model"

  val corpusPath = new Param[String](this, "corpusPath", "path to text corpus for learning")
  val corpusFormat = new Param[String](this, "corpusFormat", "dataset corpus format. txt or txtds allowed only")
  val dictPath = new Param[String](this, "dictPath", "path to dictionary of words")
  val slangPath = new Param[String](this, "slangPath", "path to custom dictionaries")
  val tokenPattern = new Param[String](this, "tokenPattern", "Regex pattern to use in tokenization of corpus. Defaults [a-zA-Z]+")

  setDefault(dictPath, "/spell/words.txt")
  setDefault(corpusFormat, "TXT")
  setDefault(tokenPattern, "[a-zA-Z]+")

  setDefault(caseSensitive, false)
  setDefault(doubleVariants, false)
  setDefault(shortCircuit, false)

  override val annotatorType: AnnotatorType = SPELL

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("SPELL"))

  def setDictPath(value: String): this.type = set(dictPath, value)

  def setCorpusFormat(value: String): this.type = set(corpusFormat, value)

  def setCorpusPath(value: String): this.type = set(corpusPath, value)

  def setSlangPath(value: String): this.type = set(slangPath, value)

  def setTokenPattern(value: String): this.type = set(tokenPattern, value)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NorvigSweetingModel = {
    val loadWords = ResourceHelper.wordCount($(dictPath), $(corpusFormat).toUpperCase, $(tokenPattern))
    val corpusWordCount =
      if (get(corpusPath).isDefined) {
        ResourceHelper.wordCount($(corpusPath), $(corpusFormat).toUpperCase, $(tokenPattern))
      } else {
      Map.empty[String, Int]
      }
    val loadSlangs = if (get(slangPath).isDefined)
      ResourceHelper.parseKeyValueText($(slangPath), $(corpusFormat).toUpperCase, ",")
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