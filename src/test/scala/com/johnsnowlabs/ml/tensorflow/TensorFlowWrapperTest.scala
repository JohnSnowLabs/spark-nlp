/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.ml.tensorflow

import org.scalatest.FlatSpec
import org.tensorflow.{AutoCloseableList, SavedModelBundle, Tensor}

class TensorFlowWrapperTest extends FlatSpec {

  "TensorFlow Wrapper" should "deserialize saved model" in {
    val tags: Array[String] = Array(SavedModelBundle.DEFAULT_TAG)
    val modelPath: String = "src/test/resources/tensorflow/models/dependency-parser/bi-lstm/"

    val model: SavedModelBundle = TensorflowWrapper.withSafeSavedModelBundleLoader(tags = tags, savedModelDir = modelPath)

    assert(model.metaGraphDef().toString.startsWith("meta_info_def"))
  }

  it should "restore session from saved model to fetch variables" in {
    val tags: Array[String] = Array(SavedModelBundle.DEFAULT_TAG)
    val modelPath: String = "src/test/resources/tensorflow/models/dependency-parser/bi-lstm/"
    val model: SavedModelBundle = TensorflowWrapper.withSafeSavedModelBundleLoader(tags = tags, savedModelDir = modelPath)

    val restoredSession = TensorflowWrapper.restoreVariablesSession(model, modelPath)

    assert(restoredSession != null)
  }

  it should "restore variable from saved model" in {
    val tags: Array[String] = Array(SavedModelBundle.DEFAULT_TAG)
    val modelPath: String = "src/test/resources/tensorflow/models/dependency-parser/bi-lstm/"
    val model: SavedModelBundle = TensorflowWrapper.withSafeSavedModelBundleLoader(tags = tags, savedModelDir = modelPath)
    val wigLstm = "bi_lstm_model/FirstBlockLSTMModule/wig_first_lstm/Read/ReadVariableOp"
    val expectedShape: Array[Long] = Array(126)

    val tensor = TensorflowWrapper.restoreVariable(model, modelPath, wigLstm)

    assert(expectedShape sameElements tensor.shape().asArray())

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
