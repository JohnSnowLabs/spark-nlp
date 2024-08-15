/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.ml.openvino

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.FileHelper
import org.scalatest.BeforeAndAfter
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID

class OpenvinoWrapperTestSpec extends AnyFlatSpec with BeforeAndAfter {
  /*
   * Dummy model was created with the following python script
    """
    import torch
    import torch.nn as nn
    import openvino

    # Define a simple neural network model
    class DummyModel(nn.Module):
      def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(in_features=10, out_features=5)

      def forward(self, x):
        return self.linear(x)

    # Create the model and dummy input
    model = DummyModel()
    dummy_input = torch.randn(1, 10)  # batch size of 1, 10 features

    # Convert the model
    ov_model = openvino.convert_model(model, example_input=dummy_input)
    # Export the model to IR format
    openvino.save_model(ov_model, './dummy_model.xml')
    """
   *
   */
  private val modelXmlPath: String =
    "src/test/resources/openvino/models/dummy_model.xml"
  private val modelBinPath: String =
    "src/test/resources/openvino/models/dummy_model.bin"

  private val tmpDirPath: String = UUID.randomUUID().toString.takeRight(12) + "_ov"
  var tmpFolder: String = _

  before {
    tmpFolder = Files
      .createDirectory(Paths.get(tmpDirPath))
      .toAbsolutePath
      .toString
  }

  after {
    FileHelper.delete(tmpFolder)
  }

  "a dummy openvino wrapper" should "compile a model correctly" taggedAs SlowTest in {
    ResourceHelper.spark.sparkContext.addFile(modelXmlPath)
    ResourceHelper.spark.sparkContext.addFile(modelBinPath)
    val openvinoWrapper = new OpenvinoWrapper(Some("dummy_model"))
    openvinoWrapper.getCompiledModel()
  }

  "a dummy openvino wrapper" should "saveToFile correctly" taggedAs SlowTest in {
    ResourceHelper.spark.sparkContext.addFile(modelXmlPath)
    ResourceHelper.spark.sparkContext.addFile(modelBinPath)
    val openvinoWrapper = new OpenvinoWrapper(Some("dummy_model"))
    openvinoWrapper.saveToFile(Paths.get(tmpFolder, "dummy_model.zip").toString)
    assert(new File(tmpFolder, "dummy_model.zip").exists())
  }
}
