package com.johnsnowlabs.nlp.annotators.cv

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.scalatest.flatspec.AnyFlatSpec

class SwinForImageClassificationTest extends AnyFlatSpec with ViTForImageClassificationBehaviors {

  behavior of "SwinForImageClassificationTest"

  val goldStandards: Map[String, String] =
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

  it should behave like
    behaviorsViTForImageClassification[SwinForImageClassification](
      SwinForImageClassification.pretrained(),
      goldStandards)
}
