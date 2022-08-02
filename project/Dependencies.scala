import sbt._

object Dependencies {

  /** ------- Spark version start ------- */
  val spark32Ver = "3.2.1"

  val is_gpu: String = System.getProperty("is_gpu", "false")
  val is_opt: String = System.getProperty("is_opt", "false")
  val is_m1: String = System.getProperty("is_m1", "false")
  val is_aarch64: String = System.getProperty("is_aarch64", "false")

  val sparkVer: String = getSparkVersion

  /** ------- Spark version end ------- */

  /** Package attributes */
  def getPackageName(is_m1: String, is_gpu: String, is_aarch64: String): String = {
    if (is_gpu.equals("true")) {
      "spark-nlp-gpu"
    } else if (is_m1.equals("true")) {
      "spark-nlp-m1"
    } else if (is_aarch64.equals("true")) {
      "spark-nlp-linux-aarch64"
    } else {
      "spark-nlp"
    }
  }

  def getSparkVersion: String = {
    spark32Ver
  }

  def getJavaTarget(is_spark32: String): String = {
    if (is_spark32.equals("true")) {
      "-target:jvm-1.8"
    } else {
      ""
    }
  }

  /** ------- Scala version start ------- */
  lazy val scala212 = "2.12.15"
  lazy val scalaVer: String = scala212

  lazy val supportedScalaVersions: Seq[String] = List(scala212)

  val scalaTestVersion = "3.2.9"

  /** ------- Scala version end ------- */

  /** ------- Dependencies start------- */

  // utilDependencies

  val typesafeVersion = "1.4.2"
  val typesafe = "com.typesafe" % "config" % typesafeVersion

  val rocksdbjniVersion = "6.29.5"
  val rocksdbjni = "org.rocksdb" % "rocksdbjni" % rocksdbjniVersion

  val awsjavasdkbundleVersion = "1.11.828"
  val awsjavasdkbundle = "com.amazonaws" % "aws-java-sdk-bundle" % awsjavasdkbundleVersion

  val liblevenshteinVersion = "3.0.0"
  val liblevenshtein = "com.github.universal-automata" % "liblevenshtein" % liblevenshteinVersion

  val greexVersion = "1.0"
  val greex = "com.navigamez" % "greex" % greexVersion

  val junitVersion = "4.13.2"
  val junit = "junit" % "junit" % junitVersion % Test

  val tensorflowVersion = "0.4.3-rc1"

  val tensorflowGPU = "com.johnsnowlabs.nlp" %% "tensorflow-gpu" % tensorflowVersion
  val tensorflowCPU = "com.johnsnowlabs.nlp" %% "tensorflow-cpu" % tensorflowVersion
  val tensorflowM1 = "com.johnsnowlabs.nlp" %% "tensorflow-m1" % tensorflowVersion
  val tensorflowLinuxAarch64 = "com.johnsnowlabs.nlp" %% "tensorflow-aarch64" % tensorflowVersion

  /** ------- Dependencies end  ------- */
}
