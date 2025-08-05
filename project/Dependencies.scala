import sbt._

object Dependencies {

  /** ------- Spark version start ------- */
  /* default spark version to base the APIS */
  val spark34Ver = "3.4.0"
  /* only used in unit tests */
  val spark30Ver = "3.0.3"
  val spark31Ver = "3.1.3"
  val spark32Ver = "3.2.3"
  val spark33Ver = "3.3.1"
  val spark35Ver = "3.5.0"

  /* required for different hardware */
  val is_gpu: String = System.getProperty("is_gpu", "false")
  val is_opt: String = System.getProperty("is_opt", "false")
  val is_silicon: String = System.getProperty("is_silicon", "false")
  val is_aarch64: String = System.getProperty("is_aarch64", "false")

  /* only used for unit tests */
  val is_spark30: String = System.getProperty("is_spark30", "false")
  val is_spark31: String = System.getProperty("is_spark31", "false")
  val is_spark32: String = System.getProperty("is_spark32", "false")
  val is_spark33: String = System.getProperty("is_spark33", "false")
  val is_spark34: String = System.getProperty("is_spark34", "true")
  val is_spark35: String = System.getProperty("is_spark35", "false")

  val sparkVer: String =
    getSparkVersion(is_spark30, is_spark31, is_spark32, is_spark33, is_spark34, is_spark35)

  /** ------- Spark version end ------- */

  /** Package attributes */
  def getPackageName(is_silicon: String, is_gpu: String, is_aarch64: String): String = {
    if (is_gpu.equals("true")) {
      "spark-nlp-gpu"
    } else if (is_silicon.equals("true")) {
      "spark-nlp-silicon"
    } else if (is_aarch64.equals("true")) {
      "spark-nlp-aarch64"
    } else {
      "spark-nlp"
    }
  }

  def getSparkVersion(
      is_spark30: String,
      is_spark31: String,
      is_spark32: String,
      is_spark33: String,
      is_spark34: String,
      is_spark35: String): String = {
    if (is_spark30.equals("true")) {
      spark30Ver
    } else if (is_spark31.equals("true")) {
      spark31Ver
    } else if (is_spark32.equals("true")) {
      spark32Ver
    } else if (is_spark33.equals("true")) {
      spark33Ver
    } else if (is_spark35.equals("true")) {
      spark35Ver
    } else {
      /* default spark version */
      spark34Ver
    }
  }

  def getJavaTarget(is_spark33: String): String = {
    if (is_spark33.equals("true")) {
      "-target:jvm-1.8"
    } else {
      ""
    }
  }

  /** ------- Scala version start ------- */
  lazy val scala212 = "2.12.15"
  lazy val scalaVer: String = scala212

  lazy val supportedScalaVersions: Seq[String] = List(scala212)

  val scalaTestVersion = "3.2.14"

  /** ------- Scala version end ------- */

  /** ------- Dependencies start------- */

  // utilDependencies

  val typesafeVersion = "1.4.2"
  val typesafe = "com.typesafe" % "config" % typesafeVersion

  val rocksdbjniVersion = "6.29.5"
  val rocksdbjni = "org.rocksdb" % "rocksdbjni" % rocksdbjniVersion

  val awsJavaSdkS3Version = "1.12.500"
  val awsJavaSdkS3 = "com.amazonaws" % "aws-java-sdk-s3" % awsJavaSdkS3Version

  val liblevenshteinVersion = "3.0.0"
  val liblevenshtein = "com.github.universal-automata" % "liblevenshtein" % liblevenshteinVersion

  val greexVersion = "1.0"
  val greex = "com.navigamez" % "greex" % greexVersion

  val junitVersion = "4.13.2"
  val junit = "junit" % "junit" % junitVersion % Test

  val tensorflowVersion = "0.4.4"
  val tensorflowGPU = "com.johnsnowlabs.nlp" %% "tensorflow-gpu" % tensorflowVersion
  val tensorflowCPU = "com.johnsnowlabs.nlp" %% "tensorflow-cpu" % tensorflowVersion
  val tensorflowM1 = "com.johnsnowlabs.nlp" %% "tensorflow-m1" % tensorflowVersion
  val tensorflowLinuxAarch64 = "com.johnsnowlabs.nlp" %% "tensorflow-aarch64" % tensorflowVersion

  val onnxRuntimeVersion = "1.19.2"
  val onnxCPU = "com.microsoft.onnxruntime" % "onnxruntime" % onnxRuntimeVersion
  val onnxGPU = "com.microsoft.onnxruntime" % "onnxruntime_gpu" % onnxRuntimeVersion

  val openVinoRuntimeVersion = "0.2.0"
  val openVinoCPU = "com.johnsnowlabs.nlp" %% "jsl-openvino-cpu" % openVinoRuntimeVersion
  val openVinoGPU = "com.johnsnowlabs.nlp" %% "jsl-openvino-gpu" % openVinoRuntimeVersion

  val gcpStorageVersion = "2.20.1"
  val gcpStorage = "com.google.cloud" % "google-cloud-storage" % gcpStorageVersion
  val azureIdentityVersion = "1.12.2"
  val azureStorageVersion = "12.26.0"
  val azureIdentity = "com.azure" % "azure-identity" % azureIdentityVersion % Provided
  val azureStorage = "com.azure" % "azure-storage-blob" % azureStorageVersion % Provided

  val llamaCppVersion = "1.0.2"
  val llamaCppCPU = "com.johnsnowlabs.nlp" % "jsl-llamacpp-cpu" % llamaCppVersion
  val llamaCppGPU = "com.johnsnowlabs.nlp" % "jsl-llamacpp-gpu" % llamaCppVersion
  val llamaCppSilicon = "com.johnsnowlabs.nlp" % "jsl-llamacpp-silicon" % llamaCppVersion
  val llamaCppAarch64 = "com.johnsnowlabs.nlp" % "jsl-llamacpp-aarch64" % llamaCppVersion

  val jsoupVersion = "1.18.2"

  val jsoup = "org.jsoup" % "jsoup" % jsoupVersion

  val jakartaMailVersion = "2.1.3"
  val jakartaMail = "jakarta.mail" % "jakarta.mail-api" % jakartaMailVersion
  val angusMailVersion = "2.0.3"
  val angusMail = "org.eclipse.angus" % "angus-mail" % angusMailVersion

  val poiVersion = "4.1.2"
  val poiDocx = "org.apache.poi" % "poi-ooxml" % poiVersion
  val scratchpad = "org.apache.poi" % "poi-scratchpad" % poiVersion

  val pdfBoxVersion = "2.0.28"
  val pdfBox = "org.apache.pdfbox" % "pdfbox" % pdfBoxVersion

  val flexmarkVersion = "0.61.34"
  val flexmark = "com.vladsch.flexmark" % "flexmark-all" % flexmarkVersion
  /** ------- Dependencies end  ------- */
}
