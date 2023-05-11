package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.TestUtils.captureOutput
import org.scalatest.flatspec.AnyFlatSpec

class HasFeaturesTestSpec extends AnyFlatSpec {
  class MockModel extends HasFeatures {
    private val protectedMockFeature =
      new StructFeature[String](this, "mockFeature").setProtected()
    def setProtectedMockFeature(value: String): this.type = set(protectedMockFeature, value)
    def getProtectedMockFeature: String = $$(protectedMockFeature)

  }

  val model = new MockModel

  behavior of "HasFeatures"

  it should "set protected params only once" taggedAs FastTest in {
    model.setProtectedMockFeature("first")
    assert(model.getProtectedMockFeature == "first")

    val output = captureOutput {
      model.setProtectedMockFeature("second")
    }
    assert(output.contains("is protected and can only be set once"))

    // should stay the same as the first value
    assert(model.getProtectedMockFeature == "first")
  }

}
