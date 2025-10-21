import Dependencies.*
import M2Resolvers.m2Resolvers
import sbtassembly.MergeStrategy

name := getPackageName(is_silicon, is_gpu, is_aarch64)

organization := "com.johnsnowlabs.nlp"

version := "6.2.0"

(ThisBuild / scalaVersion) := scalaVer

(ThisBuild / scalacOptions) += "-target:jvm-1.8"

(ThisBuild / javaOptions) += "-Xmx4096m"

(ThisBuild / javaOptions) += "-XX:+UseG1GC"

scalacOptions ++= Seq("-unchecked", "-feature", "-deprecation", "-language:implicitConversions")

(Compile / doc / scalacOptions) ++= Seq(
  "-groups",
  "-doc-title",
  "Spark NLP " + version.value + " ScalaDoc",
  "-skip-packages",
  "com.johnsnowlabs.nlp.annotator:com.johnsnowlabs.nlp.base",
  "-nowarn")

(ThisBuild / scalafmtOnCompile) := true

Compile / doc / target := baseDirectory.value / "docs/api"

// exclude memory-intensive modules from coverage
coverageExcludedPackages := ".*nlp.embeddings.*;.*ml.tensorflow.*;.*nlp.annotators.classifier.dl.*;" +
  ".*nlp.annotators.seq2seq.*;.*ml.*"

(ThisBuild / resolvers) := m2Resolvers

lazy val analyticsDependencies = Seq(
  "org.apache.spark" %% "spark-core" % sparkVer % Provided,
  "org.apache.spark" %% "spark-mllib" % sparkVer % Provided)

lazy val testDependencies = Seq(
  "org.scalatest" %% "scalatest" % scalaTestVersion % "test",
  "org.scalatest" %% "scalatest-flatspec" % scalaTestVersion % "test",
  "org.scalatest" %% "scalatest-shouldmatchers" % scalaTestVersion % "test")

lazy val utilDependencies = Seq(
  typesafe,
  rocksdbjni,
  awsJavaSdkS3
    exclude ("com.fasterxml.jackson.core", "jackson-annotations")
    exclude ("com.fasterxml.jackson.core", "jackson-databind")
    exclude ("com.fasterxml.jackson.core", "jackson-core")
    exclude ("com.fasterxml.jackson.dataformat", "jackson-dataformat-cbor")
    exclude ("commons-configuration", "commons-configuration"),
  liblevenshtein
    exclude ("com.google.guava", "guava")
    exclude ("org.apache.commons", "commons-lang3")
    exclude ("com.google.code.findbugs", "annotations")
    exclude ("org.slf4j", "slf4j-api"),
  gcpStorage
    exclude ("com.fasterxml.jackson.core", "jackson-core")
    exclude ("com.fasterxml.jackson.dataformat", "jackson-dataformat-cbor"),
  greex,
  azureIdentity,
  azureStorage,
  jsoup,
  jakartaMail,
  angusMail,
  poiDocx
    exclude ("org.apache.logging.log4j", "log4j-api"),
  scratchpad
    exclude ("org.apache.logging.log4j", "log4j-api"),
  pdfBox,
  flexmark,
  tagSoup
)

lazy val typedDependencyParserDependencies = Seq(junit)

val tensorflowDependencies: Seq[sbt.ModuleID] =
  if (is_gpu.equals("true"))
    Seq(tensorflowGPU)
  else if (is_silicon.equals("true"))
    Seq(tensorflowM1)
  else if (is_aarch64.equals("true"))
    Seq(tensorflowLinuxAarch64)
  else
    Seq(tensorflowCPU)

val onnxDependencies: Seq[sbt.ModuleID] =
  if (is_gpu.equals("true"))
    Seq(onnxGPU)
  else if (is_silicon.equals("true"))
    Seq(onnxCPU)
  else if (is_aarch64.equals("true"))
    Seq(onnxCPU)
  else
    Seq(onnxCPU)

val llamaCppDependencies =
  if (is_gpu.equals("true"))
    Seq(llamaCppGPU)
  else if (is_silicon.equals("true"))
    Seq(llamaCppSilicon)
  else if (is_aarch64.equals("true"))
    Seq(llamaCppAarch64)
  else
    Seq(llamaCppCPU)

val openVinoDependencies: Seq[sbt.ModuleID] = Seq(openVinoCPU)

lazy val mavenProps = settingKey[Unit]("workaround for Maven properties")

lazy val root = (project in file("."))
  .settings(
    crossScalaVersions := supportedScalaVersions,
    libraryDependencies ++=
      analyticsDependencies ++
        testDependencies ++
        utilDependencies ++
        tensorflowDependencies ++
        onnxDependencies ++
        llamaCppDependencies ++
        openVinoDependencies ++
        typedDependencyParserDependencies,
    // TODO potentially improve this?
    mavenProps := {
      sys.props("javacpp.platform.extension") = if (is_gpu.equals("true")) "-gpu" else ""
    })

(assembly / assemblyShadeRules) := Seq(
  ShadeRule.rename("org.apache.http.**" -> "org.apache.httpShaded@1").inAll,
  ShadeRule.rename("com.amazonaws.**" -> "com.amazonaws.ShadedByJSL@1").inAll)

(assembly / assemblyOption) := (assembly / assemblyOption).value.withIncludeScala(includeScala =
  false)

(assembly / assemblyMergeStrategy) := {
  case PathList("META-INF", "versions", "9", "module-info.class") => MergeStrategy.discard
  case PathList("module-info.class") =>
    MergeStrategy.discard // Discard any module-info.class globally
  case PathList("apache.commons.lang3", _ @_*) => MergeStrategy.discard
  case PathList("org.apache.hadoop", _ @_*) => MergeStrategy.first
  case PathList("com.amazonaws", _ @_*) => MergeStrategy.last
  case PathList("com.fasterxml.jackson") => MergeStrategy.first
  case PathList("META-INF", "io.netty.versions.properties") => MergeStrategy.first
  case PathList("org", "tensorflow", _ @_*) => MergeStrategy.first
  case x =>
    val oldStrategy = (assembly / assemblyMergeStrategy).value
    oldStrategy(x)
}

/** Test tagging start */
// Command line fast:test
lazy val FastTest = config("fast") extend Test
// Command line slow:test
lazy val SlowTest = config("slow") extend Test

configs(FastTest, SlowTest)

(Test / parallelExecution) := false
(Test / logBuffered) := false
(Test / testOptions) := Seq(Tests.Argument("-l", "com.johnsnowlabs.tags.SlowTest")) // exclude

inConfig(FastTest)(Defaults.testTasks)
(FastTest / testOptions) := Seq(Tests.Argument("-l", "com.johnsnowlabs.tags.SlowTest")) // exclude
(FastTest / parallelExecution) := false

inConfig(SlowTest)(Defaults.testTasks)
(SlowTest / testOptions) := Seq(Tests.Argument("-n", "com.johnsnowlabs.tags.SlowTest")) // include
(SlowTest / parallelExecution) := false

/** Test tagging end */

/** Enable for debugging */
(Test / testOptions) += Tests.Argument("-oF")

/** Disables tests in assembly */
(assembly / test) := {}

/** Publish test artifact * */
(Test / publishArtifact) := true

/** Copies the assembled jar to the pyspark/lib dir * */
lazy val copyAssembledJar = taskKey[Unit]("Copy assembled jar to python/lib")
lazy val copyAssembledJarForPyPi = taskKey[Unit]("Copy assembled jar to python/sparknlp/lib")

copyAssembledJar := {
  val jarFilePath = (assembly / assemblyOutputPath).value
  val newJarFilePath = baseDirectory(_ / "python" / "lib" / "sparknlp.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}

copyAssembledJarForPyPi := {
  val jarFilePath = (assembly / assemblyOutputPath).value
  val newJarFilePath = baseDirectory(_ / "python" / "sparknlp" / "lib" / "sparknlp.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}
