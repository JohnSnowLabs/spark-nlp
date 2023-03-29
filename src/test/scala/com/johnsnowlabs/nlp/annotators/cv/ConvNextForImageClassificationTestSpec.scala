package com.johnsnowlabs.nlp.annotators.cv

import org.scalatest.flatspec.AnyFlatSpec

class ConvNextForImageClassificationTestSpec
    extends AnyFlatSpec
    with ViTForImageClassificationBehaviors {

  behavior of "ConvNextForImageClassification"

  lazy val goldStandards: Map[String, String] =
    Map(
      "bluetick.jpg" -> "bluetick",
      "chihuahua.jpg" -> "Chihuahua",
      "egyptian_cat.jpeg" -> "tabby, tabby cat",
      "hen.JPEG" -> "hen",
      "hippopotamus.JPEG" -> "hippopotamus, hippo, river horse, Hippopotamus amphibius",
      "junco.JPEG" -> "junco, snowbird",
      "ostrich.JPEG" -> "ostrich, Struthio camelus",
      "ox.JPEG" -> "ox",
      "palace.JPEG" -> "palace",
      "tractor.JPEG" -> "thresher, thrasher, threshing machine")

  private lazy val model: ConvNextForImageClassification =
    ConvNextForImageClassification.pretrained()
  it should behave like
    behaviorsViTForImageClassification[ConvNextForImageClassification](
      ConvNextForImageClassification.load,
      model,
      goldStandards)
}
