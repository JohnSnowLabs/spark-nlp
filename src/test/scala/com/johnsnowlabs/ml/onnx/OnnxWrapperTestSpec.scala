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

package com.johnsnowlabs.ml.onnx

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

import java.nio.file.{Files, Path, Paths}
import java.io.File
import com.johnsnowlabs.util.FileHelper
import org.scalatest.BeforeAndAfter

import java.util.UUID

class OnnxWrapperTestSpec extends AnyFlatSpec with BeforeAndAfter {
  /*
   * Dummy model was created with the following python script
    """
    import torch
    import torch.nn as nn
    import torch.onnx

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

    # Export the model to ONNX format
    torch.onnx.export(model, dummy_input, "dummy_model.onnx", verbose=True)
    """
   *
   */
  private val modelPath: String = "src/test/resources/onnx/models/dummy_model.onnx"
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  private val tmpDirPath: String = UUID.randomUUID().toString.takeRight(12) + "_onnx"
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

  "a dummy onnx wrapper" should "get session correctly" taggedAs FastTest in {
    ResourceHelper.spark.sparkContext.addFile(modelPath)
    val onnxFileName = Some(new File(modelPath).getName)
    val dummyOnnxWrapper = new OnnxWrapper(onnxFileName, None)
    dummyOnnxWrapper.getSession(onnxSessionOptions)
  }

  "a dummy onnx wrapper" should "saveToFile correctly" taggedAs FastTest in {
    ResourceHelper.spark.sparkContext.addFile(modelPath)
    val onnxFileName = Some(new File(modelPath).getName)
    val dummyOnnxWrapper = new OnnxWrapper(onnxFileName, None)
    dummyOnnxWrapper.saveToFile(Paths.get(tmpFolder, "modelFromTest.zip").toString)
    // verify file existence
    assert(new File(tmpFolder, "modelFromTest.zip").exists())
  }

  "OnnxWrapper.read" should "not fail when loading two different files that share the same basename" taggedAs FastTest in {
    // Reproduces the SparkContext.addFile collision: on Spark 4.x, adding two different files
    // under the same registered basename within one SparkContext throws
    // "File ... exists and does not match contents of ...". This happens in practice whenever a
    // model with externally-stored ONNX weights gets loaded more than once per session (e.g. the
    // standard save-then-reload pattern), since the external data file's basename is fixed.
    val dirA = Paths.get(tmpFolder, "dirA")
    val dirB = Paths.get(tmpFolder, "dirB")
    Files.createDirectories(dirA)
    Files.createDirectories(dirB)

    val originalBytes = Files.readAllBytes(Paths.get(modelPath))
    Files.write(dirA.resolve("model.onnx"), originalBytes)
    // dirB's model.onnx has different content but the same basename as dirA's
    Files.write(dirB.resolve("model.onnx"), originalBytes ++ Array[Byte](0))

    val wrapperA =
      OnnxWrapper.read(ResourceHelper.spark, dirA.toString, zipped = false, useBundle = true)
    wrapperA.getSession(onnxSessionOptions)

    // Before the fix, this second read (different file, same basename "model.onnx") would throw
    // a SparkException from SparkContext.addFile instead of completing.
    val wrapperB =
      OnnxWrapper.read(ResourceHelper.spark, dirB.toString, zipped = false, useBundle = true)
    wrapperB.getSession(onnxSessionOptions)
  }

}
