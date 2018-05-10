package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.slf4j.LoggerFactory

class NormalizerModel(override val uid: String) extends AnnotatorModel[NormalizerModel]{

  override val annotatorType: AnnotatorType = TOKEN

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  val pattern = new Param[String](this, "pattern", "normalization regex pattern which match will be replaced with a space")

  val lowercase = new BooleanParam(this, "lowercase", "whether to convert strings to lowercase")

  protected val slangDict: MapFeature[String, String] = new MapFeature(this, "slangDict")

  private val logger = LoggerFactory.getLogger("NorvigApproach")

  def this() = this(Identifiable.randomUID("NORMALIZER"))

  def setPattern(value: String): this.type = set(pattern, value)

  def setLowerCase(value: Boolean): this.type = set(lowercase, value)

  def setSlangDict(value: Map[String, String]): this.type = set(slangDict, value)

  protected def getSlangDict: Map[String, String] = $$(slangDict)

  /** ToDo: Review implementation, Current implementation generates spaces between non-words, potentially breaking tokens*/
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] =
    annotations.map { token =>

      val cased =
        if ($(lowercase)) token.result.toLowerCase
        else token.result

      val correctedWord =
        if ($$(slangDict).contains(cased)){
          $$(slangDict)(cased)
        } else {
          cased
        }

      val nToken = correctedWord
        .replaceAll($(pattern), "")
        .trim
      Annotation(
        annotatorType,
        token.begin,
        token.end,
        nToken,
        token.metadata
      )
    }.filter(_.result.nonEmpty)

}

trait  PretrainedNormalizer {
  def pretrained(name: String = "norm_fast", language: Option[String] = Some("en"),
                 remoteLoc: String = ResourceDownloader.publicLoc): LemmatizerModel =
    ResourceDownloader.downloadModel(LemmatizerModel, name, language, remoteLoc)
}

object NormalizerModel extends ParamsAndFeaturesReadable[NormalizerModel] with PretrainedNormalizer