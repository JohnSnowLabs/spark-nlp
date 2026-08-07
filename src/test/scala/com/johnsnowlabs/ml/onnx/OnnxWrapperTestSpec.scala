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

  // The basenames used below must be unique per test (not the common default "model.onnx"):
  // FastTest suites share one JVM/SparkContext (fork := false), so registering a generic name
  // here would leak into SparkContext.addFile's own registry for the rest of the suite and could
  // break unrelated tests that happen to load a real model also named "model.onnx".
  private def uniqueModelName(): String =
    "onnx_wrapper_test_" + UUID.randomUUID().toString.take(8)

  // SparkContext.addFile only populates the driver's own copy immediately; the copy a task
  // actually reads from is fetched lazily by Executor.updateDependencies() the next time *any*
  // task runs on this SparkContext, which may well be a later, unrelated suite sharing this JVM.
  // If we deleted the source directory (e.g. via the `after` hook) before that fetch happens,
  // Spark's fetch-time content check fails and surfaces as a mismatch in whatever suite happened
  // to trigger it. So these directories must outlive the test method -- use a directory Spark
  // itself won't race to clean up, and leave it for the JVM's lifetime (like OnnxWrapper.read
  // itself does with FileUtils.forceDeleteOnExit) rather than deleting it in `after`.
  private def persistentModelDir(name: String): Path = {
    val dir = Files.createTempDirectory(name).toAbsolutePath
    import org.apache.commons.io.FileUtils
    FileUtils.forceDeleteOnExit(dir.toFile)
    dir
  }

  "OnnxWrapper.read" should "not fail when reloading the same-size model under the same basename" taggedAs FastTest in {
    // Reproduces the SparkContext.addFile collision: on Spark 4.x, adding a different file under
    // a basename that's already registered within one SparkContext throws "File ... exists and
    // does not match contents of ...". This happens in practice whenever a model with
    // externally-stored ONNX weights gets loaded more than once per session (e.g. the standard
    // save-then-reload pattern, where the reload is a re-serialization of the same architecture
    // and so has the same file size), since the external data file's basename is fixed.
    val modelName = uniqueModelName()
    val dirA = persistentModelDir("dirA")
    val dirB = persistentModelDir("dirB")

    val originalBytes = Files.readAllBytes(Paths.get(modelPath))
    Files.write(dirA.resolve(s"$modelName.onnx"), originalBytes)
    // same size as dirA's copy (bytes flipped, not appended) -- the case addFileOnce treats as
    // safe to skip re-adding, matching a reload of the same underlying model.
    val sameSizeDifferentContent = originalBytes.clone()
    sameSizeDifferentContent(0) = (sameSizeDifferentContent(0) + 1).toByte
    Files.write(dirB.resolve(s"$modelName.onnx"), sameSizeDifferentContent)

    val wrapperA = OnnxWrapper.read(
      ResourceHelper.spark,
      dirA.toString,
      zipped = false,
      useBundle = true,
      modelName = modelName)
    wrapperA.getSession(onnxSessionOptions)

    // Before the fix, this second read (different file, same basename) would throw a
    // SparkException from SparkContext.addFile instead of completing.
    val wrapperB = OnnxWrapper.read(
      ResourceHelper.spark,
      dirB.toString,
      zipped = false,
      useBundle = true,
      modelName = modelName)
    wrapperB.getSession(onnxSessionOptions)
  }

  "OnnxWrapper.wouldSkipAddFile" should "skip only when both basename and size already match" taggedAs FastTest in {
    // The safety net: a same-named file with a *different* size must be treated as a genuinely
    // different model (e.g. two unrelated annotators both defaulting to "model.onnx"), not a
    // reload of the same one -- addFileOnce must not skip re-adding it, so Spark's own mismatch
    // check gets a chance to fail loudly instead of silently serving the wrong model's weights.
    //
    // We test that decision directly against OnnxWrapper.wouldSkipAddFile rather than driving it
    // through a real OnnxWrapper.read/sc.addFile call: actually triggering Spark's mismatch
    // exception poisons the SparkContext for its remaining lifetime (see the warning on
    // addFileOnce) -- every later task on this shared, FastTest-suite-wide SparkContext would
    // start failing with the same error, well beyond this test.
    val basename = uniqueModelName() + ".onnx"
    val sc = ResourceHelper.spark.sparkContext

    assert(!OnnxWrapper.wouldSkipAddFile(sc, basename, size = 2244L))
    // same basename, same size again -- e.g. reloading the same model: safe to skip.
    assert(OnnxWrapper.wouldSkipAddFile(sc, basename, size = 2244L))
    // same basename, different size -- a genuinely different file: must not skip.
    assert(!OnnxWrapper.wouldSkipAddFile(sc, basename, size = 2245L))
  }

}
