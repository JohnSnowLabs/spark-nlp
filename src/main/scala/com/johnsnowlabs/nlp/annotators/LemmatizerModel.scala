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

trait ReadableWithPretrained extends ParamsAndFeaturesReadable[LemmatizerModel] with HasPretrained[LemmatizerModel] {
  override val defaultModelName = "lemma_antbnc"
}

object LemmatizerModel extends ReadablePretrainedTokenizer
