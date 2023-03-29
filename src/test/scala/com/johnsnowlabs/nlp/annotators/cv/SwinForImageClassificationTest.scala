package com.johnsnowlabs.nlp.annotators.cv

import org.scalatest.flatspec.AnyFlatSpec

class SwinForImageClassificationTest extends AnyFlatSpec with ViTForImageClassificationBehaviors {

  behavior of "SwinForImageClassificationTest"

  lazy val goldStandards: Map[String, String] =
    Map(
      "hen.JPEG" -> "hen",
      "chihuahua.jpg" -> "Chihuahua",
      "junco.JPEG" -> "junco, snowbird",
      "ostrich.JPEG" -> "ostrich, Struthio camelus",
      "hippopotamus.JPEG" -> "hippopotamus, hippo, river horse, Hippopotamus amphibius",
      "tractor.JPEG" -> "tractor",
      "ox.JPEG" -> "ox",
      "egyptian_cat.jpeg" -> "tabby, tabby cat",
      "bluetick.jpg" -> "bluetick",
      "palace.JPEG" -> "palace")

  private lazy val model: SwinForImageClassification = SwinForImageClassification.pretrained()
  it should behave like
    behaviorsViTForImageClassification[SwinForImageClassification](
      SwinForImageClassification.load,
      model,
      goldStandards)
}
