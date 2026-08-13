/*
 * Copyright 2017-2026 John Snow Labs
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
package com.johnsnowlabs.nlp.annotators.uncertainty

import com.johnsnowlabs.ml.onnx.OnnxWrapper
import org.apache.spark.sql.SparkSession

import java.nio.file.{Files, Paths, StandardCopyOption}

private[uncertainty] object OnnxModelStaging {

  /** Loads `<localModelPath>/model.onnx` under a caller-chosen distinct name.
    *
    * `OnnxWrapper.read(..., useBundle = true)` distributes the file with `sc.addFile` under its
    * `modelName`, and Spark's executor-side fetch cache keys purely on that basename, throwing on
    * a content mismatch. Two locally-loaded ONNX annotators in one `SparkContext` would therefore
    * collide on the default `"model.onnx"` - so each annotator stages its own copy under a name
    * of its own first.
    *
    * The copy goes to a fresh temp directory rather than next to the original: `modelSanityCheck`
    * hands back the user's own directory for a local path, and silently dropping a second full
    * copy of the model in there (potentially overwriting a file of that name) is not ours to do.
    *
    * @param localModelPath
    *   directory containing `model.onnx`, as returned by `modelSanityCheck`
    * @param uniqueModelName
    *   basename (without extension) to stage the model under, unique per annotator class
    */
  def readWithDistinctName(
      spark: SparkSession,
      localModelPath: String,
      uniqueModelName: String): OnnxWrapper = {
    val source = Paths.get(localModelPath, "model.onnx")
    require(
      Files.exists(source),
      s"Expected an ONNX model at $source. Export the model to ONNX and lay it out as " +
        "<model_dir>/model.onnx with its tokenizer files under <model_dir>/assets/.")

    val stagingDir = Files.createTempDirectory(s"${uniqueModelName}_staging")
    stagingDir.toFile.deleteOnExit()
    val staged = stagingDir.resolve(s"$uniqueModelName.onnx")
    staged.toFile.deleteOnExit()
    Files.copy(source, staged, StandardCopyOption.REPLACE_EXISTING)

    OnnxWrapper.read(
      spark,
      stagingDir.toAbsolutePath.toString,
      zipped = false,
      useBundle = true,
      modelName = uniqueModelName)
  }
}
