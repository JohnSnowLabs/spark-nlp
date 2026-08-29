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

  "ONNX CUDA preload configuration" should "prefer SparkSession runtime values" taggedAs FastTest in {
    val modeKey = "spark.jsl.settings.onnx.cuda.preload.mode"
    val pathsKey = "spark.jsl.settings.onnx.cuda.preload.paths"
    val spark = ResourceHelper.spark
    val originalMode = spark.conf.getOption(modeKey)
    val originalPaths = spark.conf.getOption(pathsKey)
    val originalSystemMode = Option(System.getProperty(modeKey))
    val originalSystemPaths = Option(System.getProperty(pathsKey))

    try {
      System.setProperty(modeKey, "search")
      System.setProperty(pathsKey, "/system/path")
      spark.conf.set(modeKey, "explicit")
      spark.conf.set(pathsKey, Seq("/runtime/a", "/runtime/b").mkString(File.pathSeparator))

      assert(
        OnnxWrapper.cudaPreloadConfig() == NativeLibraryPreloader
          .PreloadConfig(NativeLibraryPreloader.Explicit, Seq("/runtime/a", "/runtime/b")))
    } finally {
      originalMode.fold(spark.conf.unset(modeKey))(spark.conf.set(modeKey, _))
      originalPaths.fold(spark.conf.unset(pathsKey))(spark.conf.set(pathsKey, _))
      originalSystemMode.fold(System.clearProperty(modeKey))(System.setProperty(modeKey, _))
      originalSystemPaths.fold(System.clearProperty(pathsKey))(System.setProperty(pathsKey, _))
    }
  }

  it should "ignore the paths setting when preload mode is off" taggedAs FastTest in {
    val modeKey = "spark.jsl.settings.onnx.cuda.preload.mode"
    val pathsKey = "spark.jsl.settings.onnx.cuda.preload.paths"
    val spark = ResourceHelper.spark
    val originalMode = spark.conf.getOption(modeKey)
    val originalPaths = spark.conf.getOption(pathsKey)

    try {
      spark.conf.set(modeKey, "off")
      spark.conf.set(pathsKey, Seq.fill(20001)("/irrelevant").mkString(File.pathSeparator))

      assert(
        OnnxWrapper.cudaPreloadConfig() == NativeLibraryPreloader
          .PreloadConfig(NativeLibraryPreloader.Off, Seq.empty))
    } finally {
      originalMode.fold(spark.conf.unset(modeKey))(spark.conf.set(modeKey, _))
      originalPaths.fold(spark.conf.unset(pathsKey))(spark.conf.set(pathsKey, _))
    }
  }

  "ONNX option lifecycle" should "close options after a successful operation" taggedAs FastTest in {
    val options = new RecordingCloseable()

    val result = OnnxWrapper.withClosingResource(options)(_ => "created")

    assert(result == "created")
    assert(options.closed)
  }

  it should "preserve the operation failure when closing also fails" taggedAs FastTest in {
    val operationFailure = new RuntimeException("provider registration failed")
    val closeFailure = new RuntimeException("provider options close failed")
    val options = new RecordingCloseable(Some(closeFailure))

    val thrown = intercept[RuntimeException] {
      OnnxWrapper.withClosingResource(options)(_ => throw operationFailure)
    }

    assert(thrown eq operationFailure)
    assert(thrown.getSuppressed.toSeq.contains(closeFailure))
    assert(options.closed)
  }

  it should "close a successful AutoCloseable result when options closing fails" taggedAs FastTest in {
    val closeFailure = new RuntimeException("session options close failed")
    val options = new RecordingCloseable(Some(closeFailure))
    val session = new RecordingCloseable()

    val thrown = intercept[RuntimeException] {
      OnnxWrapper.withClosingResource(options)(_ => session)
    }

    assert(thrown eq closeFailure)
    assert(options.closed)
    assert(session.closed)
  }

  it should "close a partially configured option object without masking its failure" taggedAs FastTest in {
    val options = new RecordingCloseable()
    val configurationFailure = new InterruptedException("setter interrupted")

    val thrown = intercept[InterruptedException] {
      OnnxWrapper.withCloseOnFailure(options)(_ => throw configurationFailure)
    }

    assert(thrown eq configurationFailure)
    assert(options.closed)
  }

  private class RecordingCloseable(closeFailure: Option[Throwable] = None) extends AutoCloseable {
    var closed = false

    override def close(): Unit = {
      closed = true
      closeFailure.foreach(throw _)
    }
  }

}
