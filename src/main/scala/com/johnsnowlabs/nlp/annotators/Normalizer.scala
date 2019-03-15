package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.AnnotatorApproach
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/**
  * Annotator that cleans out tokens. Requires stems, hence tokens
  * @param uid required internal uid for saving annotator
  */
class Normalizer(override val uid: String) extends AnnotatorApproach[NormalizerModel] {

  override val description: String = "Cleans out tokens"
  override val outputAnnotatorType: AnnotatorType = TOKEN
  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  val cleanupPatterns = new StringArrayParam(this, "cleanupPatterns",
    "normalization regex patterns which match will be removed from token")
  val lowercase = new BooleanParam(this, "lowercase", "whether to convert strings to lowercase")
  val slangDictionary = new ExternalResourceParam(this,
    "slangDictionary", "delimited file with list of custom words to be manually corrected")

  val slangMatchCase = new BooleanParam(this, "slangMatchCase", "whether or not to be case sensitive to match slangs. Defaults to false.")

  setDefault(
    lowercase -> false,
    cleanupPatterns -> Array("[^\\pL+]"),
    slangMatchCase -> false
  )

  def getCleanupPatterns: Array[String] = $(cleanupPatterns)

  def setCleanupPatterns(value: Array[String]): this.type = set(cleanupPatterns, value)

  def getLowercase: Boolean = $(lowercase)

  def setLowercase(value: Boolean): this.type = set(lowercase, value)

  def setSlangMatchCase(value: Boolean): this.type = set(slangMatchCase, value)

  def getSlangMatchCase: Boolean = $(slangMatchCase)

  def setSlangDictionary(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "slang dictionary is a delimited text. needs 'delimiter' in options")
    set(slangDictionary, value)
  }

  def setSlangDictionary(path: String,
                         delimiter: String,
                         readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                         options: Map[String, String] = Map("format" -> "text")): this.type =
    set(slangDictionary, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))

  def this() = this(Identifiable.randomUID("NORMALIZER"))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NormalizerModel = {

    val loadSlangs = if (get(slangDictionary).isDefined) {
      val parsed = ResourceHelper.parseKeyValueText($(slangDictionary))
      if ($(slangMatchCase))
        parsed.mapValues(_.trim)
      else
        parsed.map{case (k, v) => (k.toLowerCase, v.trim)}
    }
    else
      Map.empty[String, String]

    new NormalizerModel()
      .setCleanupPatterns($(cleanupPatterns))
      .setLowercase($(lowercase))
      .setSlangDict(loadSlangs)
      .setSlangMatchCase($(slangMatchCase))
  }

}

object Normalizer extends DefaultParamsReadable[Normalizer]