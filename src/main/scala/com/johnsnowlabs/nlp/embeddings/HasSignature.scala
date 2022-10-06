package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.HasFeatures
import com.johnsnowlabs.nlp.serialization.MapFeature

trait HasSignature {
  this: HasFeatures =>

  /** It contains TF model signatures for the laded saved model
    *
    * @group param
    */
  val signatures = new MapFeature[String, String](model = this, name = "signatures")

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    if (get(signatures).isEmpty)
      set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)
}
