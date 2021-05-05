package com.johnsnowlabs.ml.tensorflow

import org.scalatest.FlatSpec
import org.tensorflow.types.TFloat32
import org.tensorflow.{SavedModelBundle, Tensor}

class TensorFlowWrapperTest extends FlatSpec {

  "TensorFlow Wrapper" should "deserialize saved model" in {
    val tags: Array[String] = Array(SavedModelBundle.DEFAULT_TAG)
    val modelPath: String = "src/test/resources/tensorflow/models/dependency-parser/bi-lstm/"

    val model: SavedModelBundle = TensorflowWrapper.withSafeSavedModelBundleLoader(tags = tags, savedModelDir = modelPath)

    assert(model != null)
  }

  it should "restore variables from saved model" in {
    val tags: Array[String] = Array(SavedModelBundle.DEFAULT_TAG)
    val modelPath: String = "src/test/resources/tensorflow/models/dependency-parser/bi-lstm/"
    val model: SavedModelBundle = TensorflowWrapper.withSafeSavedModelBundleLoader(tags = tags, savedModelDir = modelPath)
    val wigLstm = "bi_lstm_model/FirstBlockLSTMModule/wig_first_lstm/Read/ReadVariableOp"
    val expectedShape: Array[Long] = Array(126)

    val restoredSession = TensorflowWrapper.restoreVariablesSession(model, modelPath)
    //TODO: Refactor this unit test the code below should go in production code
    val outputFirstLstm = new AutoCloseableList[Tensor[_]](
      restoredSession.runner()
        .fetch(wigLstm)
        .run()
    )

    outputFirstLstm.forEach{output =>
      val tensor: Tensor[TFloat32] = output.expect(TFloat32.DTYPE)
      assert(expectedShape sameElements tensor.shape().asArray())
    }

  }

  it should "raise error when trying to restore unknown variable" in {
    val tags: Array[String] = Array(SavedModelBundle.DEFAULT_TAG)
    val modelPath: String = "src/test/resources/tensorflow/models/dependency-parser/bi-lstm/"
    val model: SavedModelBundle = TensorflowWrapper.withSafeSavedModelBundleLoader(tags = tags, savedModelDir = modelPath)
    val wigLstm = "unknownVariableName"
    val restoredSession = TensorflowWrapper.restoreVariablesSession(model, modelPath)

    assertThrows[IllegalArgumentException] {
      new AutoCloseableList[Tensor[_]](
        restoredSession.runner()
          .fetch(wigLstm)
          .run()
      )
    }

  }

}
