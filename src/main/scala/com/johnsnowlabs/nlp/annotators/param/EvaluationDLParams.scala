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

  /** Path to test dataset. If set, it is used to calculate statistics on it during training.
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

  /** Path to test dataset. If set, it is used to calculate statistics on it during training.
    *
    * @group setParam
    */
  def setTestDataset(
      path: String,
      readAs: ReadAs.Format = ReadAs.SPARK,
      options: Map[String, String] = Map("format" -> "parquet")): this.type =
    set(testDataset, ExternalResource(path, readAs, options))

  /** Path to test dataset. If set, it is used to calculate statistics on it during training.
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
