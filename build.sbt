import Dependencies._
import Resolvers.m2Resolvers
import sbtassembly.MergeStrategy

name := getPackageName(is_spark23, is_spark24, is_spark32, is_gpu)

organization := "com.johnsnowlabs.nlp"

version := "3.4.0"

(ThisBuild / scalaVersion) := scalaVer

(ThisBuild / scalacOptions) += "-target:jvm-1.8"

scalacOptions ++= Seq(
  "-unchecked",
  "-feature",
  "-language:implicitConversions"
)

(Compile / doc / scalacOptions) ++= Seq(
  "-groups",
  "-doc-title",
  "Spark NLP " + version.value + " ScalaDoc",
  "-skip-packages",
  "com.johnsnowlabs.nlp.annotator:com.johnsnowlabs.nlp.base",
  "-nowarn"
)

Compile / doc / target := baseDirectory.value / "docs/api"

// exclude memory-intensive modules from coverage
coverageExcludedPackages := ".*nlp.embeddings.*;.*ml.tensorflow.*;.*nlp.annotators.classifier.dl.*;" +
  ".*nlp.annotators.seq2seq.*"

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

(ThisBuild / resolvers) := m2Resolvers

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

sonatypeProfileName := "com.johnsnowlabs.nlp"

publishTo := sonatypePublishToBundle.value

sonatypeRepository := "https://s01.oss.sonatype.org/service/local"

sonatypeCredentialHost := "s01.oss.sonatype.org"

publishTo := {
  val nexus = "https://s01.oss.sonatype.org/"
  if (isSnapshot.value) Some("snapshots" at nexus + "content/repositories/snapshots")
  else Some("releases" at nexus + "service/local/staging/deploy/maven2")
}

homepage := Some(url("https://nlp.johnsnowlabs.com"))

scmInfo := Some(
  ScmInfo(
    url("https://github.com/JohnSnowLabs/spark-nlp"),
    "scm:git@github.com:JohnSnowLabs/spark-nlp.git"
  )
)

(ThisBuild / developers) := List(
  Developer(id = "saifjsl", name = "Saif Addin", email = "saif@johnsnowlabs.com", url = url("https://github.com/saifjsl")),
  Developer(id = "maziyarpanahi", name = "Maziyar Panahi", email = "maziyar@johnsnowlabs.com", url = url("https://github.com/maziyarpanahi")),
  Developer(id = "albertoandreottiATgmail", name = "Alberto Andreotti", email = "alberto@pacific.ai", url = url("https://github.com/albertoandreottiATgmail")),
  Developer(id = "danilojsl", name = "Danilo Burbano", email = "danilo@johnsnowlabs.com", url = url("https://github.com/danilojsl")),
  Developer(id = "rohit13k", name = "Rohit Kumar", email = "rohit@johnsnowlabs.com", url = url("https://github.com/rohit13k")),
  Developer(id = "aleksei-ai", name = "Aleksei Alekseev", email = "aleksei@pacific.ai", url = url("https://github.com/aleksei-ai")),
  Developer(id = "showy", name = "Eduardo Muñoz", email = "eduardo@johnsnowlabs.com", url = url("https://github.com/showy")),
  Developer(id = "C-K-Loan", name = "Christian Kasim Loan", email = "christian@johnsnowlabs.com", url = url("https://github.com/C-K-Loan")),
  Developer(id = "wolliq", name = "Stefano Lori", email = "stefano@johnsnowlabs.com", url = url("https://github.com/wolliq")),
  Developer(id = "vankov", name = "Ivan Vankov", email = "ivan@johnsnowlabs.com", url = url("https://github.com/vankov")),
  Developer(id = "alinapetukhova", name = "Alina Petukhova", email = "alina@johnsnowlabs.com", url = url("https://github.com/alinapetukhova")),
  Developer(id = "hatrungduc", name = "Devin Ha", email = "trung@johnsnowlabs.com", url = url("https://github.com/hatrungduc"))
)

lazy val analyticsDependencies = Seq(
  "org.apache.spark" %% "spark-core" % sparkVer % Provided,
  "org.apache.spark" %% "spark-mllib" % sparkVer % Provided
)

lazy val testDependencies = Seq(
  "org.scalatest" %% "scalatest" % scalaTestVersion % "test",
  "org.scalatest" %% "scalatest-flatspec" % scalaTestVersion % "test",
  "org.scalatest" %% "scalatest-shouldmatchers" % scalaTestVersion % "test"
)

lazy val utilDependencies = Seq(
  typesafe,
  rocksdbjni,
  awsjavasdkbundle
    exclude("com.fasterxml.jackson.core", "jackson-annotations")
    exclude("com.fasterxml.jackson.core", "jackson-databind")
    exclude("com.fasterxml.jackson.core", "jackson-core")
    exclude("commons-configuration", "commons-configuration"),
  liblevenshtein
    exclude("com.google.guava", "guava")
    exclude("org.apache.commons", "commons-lang3"),
  greex,
  json4s
)

lazy val typedDependencyParserDependencies = Seq(
  trove4j,
  junit
)

val tensorflowDependencies: Seq[sbt.ModuleID] =
  if (is_gpu.equals("true"))
    Seq(tensorflowGPU)
  else
    Seq(tensorflowCPU)

val pytorchDependencies: Seq[sbt.ModuleID] =
  if (is_gpu.equals("true")) {
    Seq(
      pytorchCPU
    )
  } else {
    Seq(
      pytorchCPU
    )
  }

lazy val mavenProps = settingKey[Unit]("workaround for Maven properties")

lazy val root = (project in file("."))
  .settings(
    crossScalaVersions := supportedScalaVersions,
    libraryDependencies ++=
      analyticsDependencies ++
        testDependencies ++
        utilDependencies ++
        tensorflowDependencies ++
        typedDependencyParserDependencies ++
        pytorchDependencies,
    // TODO potentially improve this?
    mavenProps := {
      sys.props("javacpp.platform.extension") = if (is_gpu.equals("true")) "-gpu" else ""
    }
  )

(assembly / assemblyShadeRules) := Seq(
  ShadeRule.rename("org.apache.http.**" -> "org.apache.httpShaded@1").inAll,
  ShadeRule.rename("com.amazonaws.**" -> "com.amazonaws.ShadedByJSL@1").inAll
)

(assembly / assemblyOption) := (assembly / assemblyOption).value.copy(
  includeScala = false
)

(assembly / assemblyMergeStrategy) := {
  case PathList("META-INF", "versions", "9", "module-info.class") => MergeStrategy.discard
  case PathList("META-INF", "io.netty.versions.properties") => MergeStrategy.first
  case PathList("apache.commons.lang3", _@_*) => MergeStrategy.discard
  case PathList("com.amazonaws", _@_*) => MergeStrategy.last
  case PathList("com.fasterxml.jackson") => MergeStrategy.first
  case PathList("com", "google", xs@_*) => MergeStrategy.last
  case PathList("com", "googlecode", xs@_*) => MergeStrategy.last
  case PathList("org.apache.hadoop", _@_*) => MergeStrategy.first
  case PathList("org", "tensorflow", _@_*) => MergeStrategy.first
  case PathList("org.apache.commons", _@_*) => MergeStrategy.first
  case PathList("org", "slf4j", xs@_*) => MergeStrategy.first
  case PathList("ai", "djl", _@_*) => MergeStrategy.first
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
lazy val copyAssembledJar = taskKey[Unit]("Copy assembled jar to pyspark/lib")
lazy val copyAssembledJarForPyPi = taskKey[Unit]("Copy assembled jar to pyspark/sparknlp/lib")

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