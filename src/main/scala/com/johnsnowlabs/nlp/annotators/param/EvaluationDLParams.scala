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

package com.johnsnowlabs.nlp.annotators.param

import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.param._

trait EvaluationDLParams extends Params {

  /** Level of verbosity during training (Default: `Verbose.Silent.id`)
    *
    * @group param
    */
  val verbose = new IntParam(this, "verbose", "Level of verbosity during training")

  /** Choose the proportion of training dataset to be validated against the model on each Epoch
    * (Default: `0.0f`). The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
    *
    * @group param
    */
  val validationSplit = new FloatParam(
    this,
    "validationSplit",
    "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.")

  /** Whether logs for validation to be extended (Default: `false`): it displays time and
    * evaluation of each label
    *
    * @group param
    */
  val evaluationLogExtended = new BooleanParam(
    this,
    "evaluationLogExtended",
    "Whether logs for validation to be extended: it displays time and evaluation of each label. Default is false.")

  /** Whether to output to annotators log folder (Default: `false`)
    *
    * @group param
    */
  val enableOutputLogs =
    new BooleanParam(this, "enableOutputLogs", "Whether to output to annotators log folder")

  /** Folder path to save training logs (Default: `""`)
    *
    * @group param
    */
  val outputLogsPath =
    new Param[String](this, "outputLogsPath", "Folder path to save training logs")

  /** Path to a parquet file of a test dataset. If set, it is used to calculate statistics on it
    * during training.
    *
    * @group param
    */
  val testDataset = new ExternalResourceParam(
    this,
    "testDataset",
    "Path to test dataset. If set, it is used to calculate statistics on it during training.")

  /** Level of verbosity during training (Default: `Verbose.Silent.id`)
    *
    * @group setParam
    */
  def setVerbose(verbose: Int): this.type = set(this.verbose, verbose)

  /** Level of verbosity during training (Default: `Verbose.Silent.id`)
    *
    * @group setParam
    */
  def setVerbose(verbose: Verbose.Level): this.type =
    set(this.verbose, verbose.id)

  /** Choose the proportion of training dataset to be validated against the model on each Epoch
    * (Default: `0.0f`). The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
    *
    * @group setParam
    */
  def setValidationSplit(validationSplit: Float): this.type =
    set(this.validationSplit, validationSplit)

  /** Whether logs for validation to be extended: it displays time and evaluation of each label.
    * Default is false.
    *
    * @group setParam
    */
  def setEvaluationLogExtended(evaluationLogExtended: Boolean): this.type =
    set(this.evaluationLogExtended, evaluationLogExtended)

  /** Whether to output to annotators log folder (Default: `false`)
    *
    * @group setParam
    */
  def setEnableOutputLogs(enableOutputLogs: Boolean): this.type =
    set(this.enableOutputLogs, enableOutputLogs)

  /** Folder path to save training logs (Default: `""`)
    *
    * @group setParam
    */
  def setOutputLogsPath(path: String): this.type =
    set(this.outputLogsPath, path)

  /** Path to a parquet file of a test dataset. If set, it is used to calculate statistics on it
    * during training.
    *
    * The parquet file must be a dataframe that has the same columns as the model that is being
    * trained. For example, if the model needs as input `DOCUMENT`, `TOKEN`, `WORD_EMBEDDINGS`
    * (Features) and `NAMED_ENTITY` (label) then these columns also need to be present while
    * saving the dataframe. The pre-processing steps for the training dataframe should also be
    * applied to the test dataframe.
    *
    * An example on how to create such a parquet file could be:
    *
    * {{{
    * // assuming preProcessingPipeline
    * val Array(train, test) = data.randomSplit(Array(0.8, 0.2))
    *
    * preProcessingPipeline
    *   .fit(test)
    *   .transform(test)
    *   .write
    *   .mode("overwrite")
    *   .parquet("test_data")
    *
    * annotator.setTestDataset("test_data")
    * }}}
    *
    * @group setParam
    */
  def setTestDataset(
      path: String,
      readAs: ReadAs.Format = ReadAs.SPARK,
      options: Map[String, String] = Map("format" -> "parquet")): this.type =
    set(testDataset, ExternalResource(path, readAs, options))

  /** ExternalResource to a parquet file of a test dataset. If set, it is used to calculate
    * statistics on it during training.
    *
    * When using an ExternalResource, only parquet files are accepted for this function.
    *
    * The parquet file must be a dataframe that has the same columns as the model that is being
    * trained. For example, if the model needs as input `DOCUMENT`, `TOKEN`, `WORD_EMBEDDINGS`
    * (Features) and `NAMED_ENTITY` (label) then these columns also need to be present while
    * saving the dataframe. The pre-processing steps for the training dataframe should also be
    * applied to the test dataframe.
    *
    * An example on how to create such a parquet file could be:
    *
    * {{{
    * // assuming preProcessingPipeline
    * val Array(train, test) = data.randomSplit(Array(0.8, 0.2))
    *
    * preProcessingPipeline
    *   .fit(test)
    *   .transform(test)
    *   .write
    *   .mode("overwrite")
    *   .parquet("test_data")
    *
    * annotator.setTestDataset("test_data")
    * }}}
    *
    * @group setParam
    */
  def setTestDataset(er: ExternalResource): this.type = set(testDataset, er)

  /** Choose the proportion of training dataset to be validated against the model on each Epoch
    * (Default: `0.0f`). The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
    *
    * @group getParam
    */
  def getValidationSplit: Float = $(this.validationSplit)

  /** Whether to output to annotators log folder (Default: `false`)
    *
    * @group getParam
    */
  def getEnableOutputLogs: Boolean = $(enableOutputLogs)

  /** Folder path to save training logs (Default: `""`)
    *
    * @group getParam
    */
  def getOutputLogsPath: String = $(outputLogsPath)

  setDefault(
    verbose -> Verbose.Silent.id,
    validationSplit -> 0.0f,
    enableOutputLogs -> false,
    evaluationLogExtended -> false,
    outputLogsPath -> "")

}
