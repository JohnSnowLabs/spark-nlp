import sbt._

object Dependencies {

  /** ------- Spark version start ------- */
  val spark23Ver = "2.3.4"
  val spark24Ver = "2.4.7"
  val spark30Ver = "3.0.2"
  val spark32Ver = "3.2.0"

  val is_gpu: String = System.getProperty("is_gpu", "false")
  val is_opt: String = System.getProperty("is_opt", "false")
  val is_spark23: String = System.getProperty("is_spark23", "false")
  val is_spark24: String = System.getProperty("is_spark24", "false")
  val is_spark30: String = System.getProperty("is_spark30", "false")
  val is_spark32: String = System.getProperty("is_spark32", "false")

  val sparkVer: String = getSparkVersion(is_spark23, is_spark24, is_spark32)

  /** ------- Spark version end ------- */


  /** Package attributes */
  def getPackageName(is_spark23: String, is_spark24: String, is_spark32: String, is_gpu: String): String = {
    if (is_gpu.equals("true") && is_spark23.equals("true")) {
      "spark-nlp-gpu-spark23"
    } else if (is_gpu.equals("true") && is_spark24.equals("true")) {
      "spark-nlp-gpu-spark24"
    } else if (is_gpu.equals("true") && is_spark32.equals("true")) {
      "spark-nlp-gpu-spark32"
    } else if (is_gpu.equals("true") && is_spark32.equals("false")) {
      "spark-nlp-gpu"
    } else if (is_gpu.equals("false") && is_spark23.equals("true")) {
      "spark-nlp-spark23"
    } else if (is_gpu.equals("false") && is_spark24.equals("true")) {
      "spark-nlp-spark24"
    } else if (is_gpu.equals("false") && is_spark32.equals("true")) {
      "spark-nlp-spark32"
    } else {
      "spark-nlp"
    }
  }

  def getSparkVersion(is_spark23: String, is_spark24: String, is_spark32: String): String = {
    if (is_spark24 == "true") spark24Ver
    else if (is_spark23 == "true") spark23Ver
    else if (is_spark32 == "true") spark32Ver
    else spark30Ver
  }

  def getJavaTarget(is_spark23: String, is_spark24: String): String = {
    if (is_spark24.equals("true") || is_spark23.equals("true")) {
      "-target:jvm-1.8"
    } else {
      ""
    }
  }

  /** ------- Scala version start ------- */
  lazy val scala211 = "2.11.12"
  lazy val scala212 = "2.12.10"
  lazy val scalaVer: String = if (is_spark23 == "true" | is_spark24 == "true") scala211 else scala212

  lazy val supportedScalaVersions = List(scala212, scala211)

  val scalaTestVersion = "3.2.9"

  /** ------- Scala version end ------- */

  /** ------- Dependencies start------- */

  // utilDependencies

  val typesafeVersion = "1.4.1"
  val typesafe = "com.typesafe" % "config" % typesafeVersion

  val rocksdbjniVersion = "6.5.3"
  val rocksdbjni = "org.rocksdb" % "rocksdbjni" % rocksdbjniVersion

  val awsjavasdkbundleVersion = "1.11.603"
  val awsjavasdkbundle = "com.amazonaws" % "aws-java-sdk-bundle" % awsjavasdkbundleVersion

  val liblevenshteinVersion = "3.0.0"
  val liblevenshtein = "com.github.universal-automata" % "liblevenshtein" % liblevenshteinVersion

  val greexVersion = "1.0"
  val greex = "com.navigamez" % "greex" % greexVersion

  val json4sVersion: String = if (is_spark32 == "true") "3.7.0-M11" else "3.5.3"

  val json4s = "org.json4s" %% "json4s-ext" % json4sVersion

  val trove4jVersion = "3.0.3"
  val trove4j = "net.sf.trove4j" % "trove4j" % trove4jVersion

  val junitVersion = "4.13.2"
  val junit = "junit" % "junit" % junitVersion % Test

  val tensorflowGPUVersion = "0.3.3"
  val tensorflowGPU = "com.johnsnowlabs.nlp" %% "tensorflow-gpu" % tensorflowGPUVersion

  val tensorflowCPUVersion = "0.3.3"
  val tensorflowCPU = "com.johnsnowlabs.nlp" %% "tensorflow-cpu" % tensorflowCPUVersion

  val pytorchCPUVersion = "0.14.0"
  val pytorchCPU = "ai.djl.pytorch" % "pytorch-engine" % pytorchCPUVersion

//  val pytorchGPUVersion = "1.9.1"
//  val pytorchGPU = "ai.djl.pytorch" % "pytorch-native-cu111" % pytorchGPUVersion

  /** ------- Dependencies end  ------- */
}