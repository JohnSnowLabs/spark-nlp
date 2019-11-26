package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, ParamsAndFeaturesReadable}
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.ml.util.Identifiable

class LemmatizerModel(override val uid: String) extends AnnotatorModel[LemmatizerModel] {

  override val outputAnnotatorType: AnnotatorType = TOKEN

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  val lemmaDict: MapFeature[String, String] = new MapFeature(this, "lemmaDict")

  def this() = this(Identifiable.randomUID("LEMMATIZER"))

  def setLemmaDict(value: Map[String, String]): this.type = set(lemmaDict, value)

  /**
    * @return one to one annotation from token to a lemmatized word, if found on dictionary or leave the word as is
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map { tokenAnnotation =>
      val token = tokenAnnotation.result
      Annotation(
        outputAnnotatorType,
        tokenAnnotation.begin,
        tokenAnnotation.end,
        $$(lemmaDict).getOrElse(token, token),
        tokenAnnotation.metadata
      )
    }
  }

}

trait ReadablePretrainedLemmatizer extends ParamsAndFeaturesReadable[LemmatizerModel] with HasPretrained[LemmatizerModel] {
  override val defaultModelName = Some("lemma_antbnc")

  /** Java compliant-overrides */
  override def pretrained(): LemmatizerModel = super.pretrained()
  override def pretrained(name: String): LemmatizerModel = super.pretrained(name)
  override def pretrained(name: String, lang: String): LemmatizerModel = super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): LemmatizerModel = super.pretrained(name, lang, remoteLoc)
}

object LemmatizerModel extends ReadablePretrainedLemmatizer
