import sbt._

object Dependencies {

  /** ------- Spark version start ------- */
  final case class SparkBuildProfile(
      id: String,
      compileBaseline: String,
      artifactSuffix: String,
      supportedVersions: Seq[String])

  final case class SparkBuildVariant(id: String, artifactBaseName: String) {
    def isGpu: Boolean = id == "gpu"
  }

  val spark400Ver = "4.0.0"
  val spark401Ver = "4.0.1"
  val spark410Ver = "4.1.0"
  val spark411Ver = "4.1.1"
  val spark412Ver = "4.1.2"

  val spark400Profile = SparkBuildProfile(
    id = "spark400",
    compileBaseline = spark400Ver,
    artifactSuffix = "-spark400",
    supportedVersions = Seq(spark400Ver))

  val spark4Profile = SparkBuildProfile(
    id = "spark4",
    compileBaseline = spark401Ver,
    artifactSuffix = "",
    supportedVersions = Seq(spark401Ver, spark410Ver, spark411Ver, spark412Ver))

  val sparkBuildProfiles: Map[String, SparkBuildProfile] =
    Seq(spark400Profile, spark4Profile).map(profile => profile.id -> profile).toMap

  val sparkBuildProfileId: String =
    System.getProperty("spark.build.profile", spark4Profile.id).trim.toLowerCase

  val sparkBuildProfile: SparkBuildProfile = sparkBuildProfiles.getOrElse(
    sparkBuildProfileId,
    throw new IllegalArgumentException(
      s"Unsupported Spark build profile '$sparkBuildProfileId'. " +
        s"Supported profiles: ${sparkBuildProfiles.keys.toSeq.sorted.mkString(", ")}"))

  val cpuVariant = SparkBuildVariant("cpu", "spark-nlp")
  val gpuVariant = SparkBuildVariant("gpu", "spark-nlp-gpu")
  val siliconVariant = SparkBuildVariant("silicon", "spark-nlp-silicon")
  val aarch64Variant = SparkBuildVariant("aarch64", "spark-nlp-aarch64")

  val sparkBuildVariants: Map[String, SparkBuildVariant] =
    Seq(cpuVariant, gpuVariant, siliconVariant, aarch64Variant)
      .map(variant => variant.id -> variant)
      .toMap

  val sparkBuildVariantId: String =
    System.getProperty("spark.build.variant", cpuVariant.id).trim.toLowerCase

  val sparkBuildVariant: SparkBuildVariant = sparkBuildVariants.getOrElse(
    sparkBuildVariantId,
    throw new IllegalArgumentException(
      s"Unsupported Spark build variant '$sparkBuildVariantId'. " +
        s"Supported variants: ${sparkBuildVariants.keys.toSeq.sorted.mkString(", ")}"))

  private val sparkVersionOverride = System.getProperty("spark.version", "").trim

  val sparkVer: String =
    if (sparkVersionOverride.nonEmpty) sparkVersionOverride
    else sparkBuildProfile.compileBaseline

  require(
    sparkBuildProfile.supportedVersions.contains(sparkVer),
    s"Spark version '$sparkVer' is not supported by build profile '${sparkBuildProfile.id}'. " +
      s"Supported versions: ${sparkBuildProfile.supportedVersions.mkString(", ")}")

  val sparkArtifactBaseName: String =
    sparkBuildVariant.artifactBaseName + sparkBuildProfile.artifactSuffix

  val isPublishBaseline: Boolean = sparkVer == sparkBuildProfile.compileBaseline

  /** ------- Spark version end ------- */

  /** ------- Scala version start ------- */
  lazy val scalaVer: String = "2.13.16"

  lazy val supportedScalaVersions: Seq[String] = Seq(scalaVer)

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

  val onnxRuntimeVersion = "1.23.0"
  val onnxCPU = "com.microsoft.onnxruntime" % "onnxruntime" % onnxRuntimeVersion
  val onnxGPU = "com.microsoft.onnxruntime" % "onnxruntime_gpu" % onnxRuntimeVersion

  val openVinoRuntimeVersion = "0.2.0"
  val openVinoCPU = "com.johnsnowlabs.nlp" % "jsl-openvino-cpu_2.12" % openVinoRuntimeVersion
  val openVinoGPU = "com.johnsnowlabs.nlp" % "jsl-openvino-gpu_2.12" % openVinoRuntimeVersion
  val openVinoSilicon = "com.johnsnowlabs.nlp" % "jsl-openvino-silicon_2.12" % openVinoRuntimeVersion

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
