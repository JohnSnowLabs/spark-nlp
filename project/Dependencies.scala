import sbt._

object Dependencies {

  /** ------- Spark version start ------- */
  val spark400Ver = "4.0.0"
  val spark401Ver = "4.0.1"
  val spark410Ver = "4.1.0"
  val spark411Ver = "4.1.1"
  val spark412Ver = "4.1.2"

  val spark4Versions: Seq[String] = Seq(
    spark400Ver,
    spark401Ver,
    spark410Ver,
    spark411Ver,
    spark412Ver)

  /* Default Spark 4 baseline. Use -Dspark.version=<version> for a specific Spark 4.x lane. */
  val spark4DefaultVer = spark401Ver

  /* required for different hardware */
  val is_gpu: String = System.getProperty("is_gpu", "false")
  val is_opt: String = System.getProperty("is_opt", "false")
  val is_silicon: String = System.getProperty("is_silicon", "false")
  val is_aarch64: String = System.getProperty("is_aarch64", "false")

  /* only used for unit tests and compatibility lanes */
  val is_spark400: String = System.getProperty("is_spark400", "false")
  val is_spark401: String = System.getProperty("is_spark401", "false")
  val is_spark410: String = System.getProperty("is_spark410", "false")
  val is_spark411: String = System.getProperty("is_spark411", "false")
  val is_spark412: String = System.getProperty("is_spark412", "false")

  private val sparkVersionOverride = System.getProperty("spark.version", "").trim

  private val sparkProfiles = Seq(
    "-Dis_spark400=true" -> spark400Ver,
    "-Dis_spark401=true" -> spark401Ver,
    "-Dis_spark410=true" -> spark410Ver,
    "-Dis_spark411=true" -> spark411Ver,
    "-Dis_spark412=true" -> spark412Ver)

  private val selectedSparkProfiles = sparkProfiles
    .zip(
      Seq(
        is_spark400,
        is_spark401,
        is_spark410,
        is_spark411,
        is_spark412))
    .collect { case ((profile, version), "true") => profile -> version }

  require(
    selectedSparkProfiles.size <= 1,
    s"Select at most one Spark 4 profile: ${sparkProfiles.map(_._1).mkString(", ")}")

  require(
    sparkVersionOverride.isEmpty || selectedSparkProfiles.isEmpty,
    "Use either -Dspark.version=<Spark 4.x version> or a Spark profile flag, not both")

  val sparkVer: String = getSparkVersion(sparkVersionOverride, selectedSparkProfiles)

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
      sparkVersionOverride: String,
      selectedSparkProfiles: Seq[(String, String)]): String = {
    val selectedVersion =
      if (sparkVersionOverride.nonEmpty) sparkVersionOverride
      else selectedSparkProfiles.headOption.map(_._2).getOrElse(spark4DefaultVer)

    require(
      spark4Versions.contains(selectedVersion),
      s"Unsupported Spark version '$selectedVersion'. Supported Spark 4 versions: ${spark4Versions.mkString(", ")}")

    selectedVersion
  }

  /** ------- Scala version start ------- */
  lazy val scalaVer: String = "2.13.16"

  lazy val supportedScalaVersions: Seq[String] = List(scalaVer)

  val scalaTestVersion = "3.2.19"

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

  val onnxRuntimeVersion = "1.24.3"
  val onnxCPU = "com.microsoft.onnxruntime" % "onnxruntime" % onnxRuntimeVersion
  val onnxGPU = "com.microsoft.onnxruntime" % "onnxruntime_gpu" % onnxRuntimeVersion

  val openVinoRuntimeVersion = "0.2.0"
  val openVinoCPU = "com.johnsnowlabs.nlp" % "jsl-openvino-cpu_2.12" % openVinoRuntimeVersion
  val openVinoGPU = "com.johnsnowlabs.nlp" % "jsl-openvino-gpu_2.12" % openVinoRuntimeVersion

  val gcpStorageVersion = "2.20.1"
  val gcpStorage = "com.google.cloud" % "google-cloud-storage" % gcpStorageVersion
  val azureIdentityVersion = "1.12.2"
  val azureStorageVersion = "12.26.0"
  val azureIdentity = "com.azure" % "azure-identity" % azureIdentityVersion % Provided
  val azureStorage = "com.azure" % "azure-storage-blob" % azureStorageVersion % Provided

  val llamaCppVersion = "2.0.3"
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

  val poiFullVersion = "5.4.1"
  val poiSchemas = "org.apache.poi" % "poi-ooxml-full" % poiFullVersion
  val poiDocx = "org.apache.poi" % "poi-ooxml" % poiFullVersion
  val scratchpad = "org.apache.poi" % "poi-scratchpad" % poiFullVersion

  val pdfBoxVersion = "2.0.28"
  val pdfBox = "org.apache.pdfbox" % "pdfbox" % pdfBoxVersion

  val flexmarkVersion = "0.61.34"
  val flexmark = "com.vladsch.flexmark" % "flexmark-all" % flexmarkVersion

  val tagSoupVersion = "1.2.1"
  val tagSoup = "org.ccil.cowan.tagsoup" % "tagsoup" % tagSoupVersion

  val pineconeScalaClient = "io.cequence" %% "pinecone-scala-client" % "1.3.2"


  val json4sVersion = "4.0.7"
  val json4sNative = "org.json4s" %% "json4s-native" % json4sVersion

  val scalaParallelCollectionsVersion = "1.2.0"
  val scalaParallelCollections =
    "org.scala-lang.modules" %% "scala-parallel-collections" % scalaParallelCollectionsVersion
  /** ------- Dependencies end  ------- */
}
