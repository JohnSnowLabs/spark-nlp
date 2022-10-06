package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.HasFeatures
import com.johnsnowlabs.nlp.serialization.MapFeature

trait HasVocabulary {
  this: HasFeatures =>

  /** @group setParam */
  def sentenceStartTokenId: Int = {
    $$(vocabulary)("[CLS]")
  }

  /** @group setParam */
  def sentenceEndTokenId: Int = {
    $$(vocabulary)("[SEP]")
  }

  /** Vocabulary used to encode the words to ids with WordPieceEncoder
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary")

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)
}
