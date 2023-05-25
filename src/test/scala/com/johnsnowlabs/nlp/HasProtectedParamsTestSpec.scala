package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.TestUtils.captureOutput
import org.apache.spark.ml.param.Param
import org.scalatest.flatspec.AnyFlatSpec
class HasProtectedParamsTestSpec extends AnyFlatSpec {
  class MockModel extends AnnotatorModel[MockModel] with HasProtectedParams {
    override val uid: String = "MockModel"
    override val outputAnnotatorType: AnnotatorType = AnnotatorType.DUMMY
    override val inputAnnotatorTypes: Array[AnnotatorType] = Array(AnnotatorType.DUMMY)

    val protectedParam =
      new Param[String](this, "MockString", "Mock protected Param").setProtected()
    def setProtectedParam(value: String): this.type = {
      set(protectedParam, value)
    }

    def getProtectedParam: String = {
      $(protectedParam)
    }

  }

  val model = new MockModel

  behavior of "HasProtectedParams"

  it should "set protected params only once" taggedAs FastTest in {
    model.setProtectedParam("first")

    assert(model.getProtectedParam == "first")

    val output = captureOutput {
      model.setProtectedParam("second")

    }
    assert(output.contains("is protected and can only be set once"))
    assert(model.getProtectedParam == "first")
  }
}
