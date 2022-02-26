import sbt._

object Dependencies {

  /** ------- Spark version start ------- */
  // Spark 3.0.x and 3.1.x are similar
  val spark30Ver = "3.1.3"
  val spark32Ver = "3.2.1"

  val is_gpu: String = System.getProperty("is_gpu", "false")
  val is_opt: String = System.getProperty("is_opt", "false")
  val is_spark30: String = System.getProperty("is_spark30", "false")
  val is_spark32: String = System.getProperty("is_spark32", "false")

  val sparkVer: String = getSparkVersion(is_spark30)

  /** ------- Spark version end ------- */

  /** Package attributes */
  def getPackageName(is_spark30: String, is_gpu: String): String = {
    if (is_gpu.equals("true") && is_spark30.equals("true")) {
      "spark-nlp-gpu-spark30"
    } else if (is_gpu.equals("true") && is_spark30.equals("false")) {
      "spark-nlp-gpu"
    } else if (is_gpu.equals("false") && is_spark30.equals("true")) {
      "spark-nlp-spark30"
    } else {
      "spark-nlp"
    }
  }

  def getSparkVersion(is_spark30: String): String = {
    if (is_spark30 == "true") spark30Ver
    else spark32Ver
  }

  def getJavaTarget(is_spark30: String, is_spark32: String): String = {
    if (is_spark30.equals("true") || is_spark32.equals("true")) {
      "-target:jvm-1.8"
    } else {
      ""
    }
  }

  /** ------- Scala version start ------- */
  lazy val scala212 = "2.12.10"
  lazy val scalaVer: String = scala212

  lazy val supportedScalaVersions: Seq[String] = List(scala212)

  val scalaTestVersion = "3.2.9"

  /** ------- Scala version end ------- */

  /** ------- Dependencies start------- */

  // utilDependencies

  val typesafeVersion = "1.4.2"
  val typesafe = "com.typesafe" % "config" % typesafeVersion

  val rocksdbjniVersion = "6.5.3"
  val rocksdbjni = "org.rocksdb" % "rocksdbjni" % rocksdbjniVersion

  val awsjavasdkbundleVersion = "1.11.603"
  val awsjavasdkbundle = "com.amazonaws" % "aws-java-sdk-bundle" % awsjavasdkbundleVersion

  val liblevenshteinVersion = "3.0.0"
  val liblevenshtein = "com.github.universal-automata" % "liblevenshtein" % liblevenshteinVersion

  val greexVersion = "1.0"
  val greex = "com.navigamez" % "greex" % greexVersion

  val json4sVersion: String = if (is_spark30 == "true") "3.7.0-M5" else "3.7.0-M11"
  val json4s = "org.json4s" %% "json4s-ext" % json4sVersion

  val junitVersion = "4.13.2"
  val junit = "junit" % "junit" % junitVersion % Test

  val tensorflowGPUVersion = "0.4.0"
  val tensorflowGPU = "com.johnsnowlabs.nlp" %% "tensorflow-gpu" % tensorflowGPUVersion

  val tensorflowCPUVersion = "0.4.0"
  val tensorflowCPU = "com.johnsnowlabs.nlp" %% "tensorflow-cpu" % tensorflowCPUVersion

  /** ------- Dependencies end  ------- */
}
