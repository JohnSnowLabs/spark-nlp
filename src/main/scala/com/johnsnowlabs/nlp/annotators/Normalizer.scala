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

  val patterns = new StringArrayParam(this, "patterns",
    "normalization regex patterns which match will be replaced with a space")
  val lowercase = new BooleanParam(this, "lowercase", "whether to convert strings to lowercase")
  val slangDictionary = new ExternalResourceParam(this,
    "slangDictionary", "delimited file with list of custom words to be manually corrected")

  setDefault(patterns, Array("[^\\pL+]"))
  setDefault(lowercase, false)

  def getPatterns: Array[String] = $(patterns)

  def setPatterns(value: Array[String]): this.type = set(patterns, value)

  def getLowercase: Boolean = $(lowercase)

  def setLowercase(value: Boolean): this.type = set(lowercase, value)

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

    val loadSlangs = if (get(slangDictionary).isDefined)
      ResourceHelper.parseKeyValueText($(slangDictionary))
    else
      Map.empty[String, String]

    new NormalizerModel()
      .setPatterns($(patterns))
      .setLowercase($(lowercase))
      .setSlangDict(loadSlangs)
  }

}

object Normalizer extends DefaultParamsReadable[Normalizer]