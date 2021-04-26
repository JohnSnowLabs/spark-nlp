import Dependencies._
import Resolvers.m2Resolvers
import sbtassembly.MergeStrategy

name:= getPackageName(is_spark23, is_spark24, is_gpu)

organization:= "com.johnsnowlabs.nlp"

version := "3.0.2"

scalaVersion in ThisBuild := scalaVer

scalacOptions in ThisBuild += "-target:jvm-1.8"

scalacOptions ++= Seq(
  "-unchecked",
  "-feature",
  "-language:implicitConversions"
)

scalacOptions in (Compile, doc) ++= Seq(
  "-groups",
  "-doc-title",
  "Spark NLP " + version.value + " ScalaDoc",
  "-skip-packages",
  "com.johnsnowlabs.nlp.annotator:com.johnsnowlabs.nlp.base",
  "-nowarn"
)

target in Compile in doc := baseDirectory.value / "docs/api"

licenses  += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

resolvers in ThisBuild := m2Resolvers

assemblyShadeRules in assembly := Seq(
  ShadeRule.rename("org.apache.http.**" -> "org.apache.httpShaded@1").inAll,
  ShadeRule.rename("com.amazonaws.**" -> "com.amazonaws.ShadedByJSL@1").inAll
)

assemblyOption in assembly := (assemblyOption in assembly).value.copy(
  includeScala = false
)

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

/** Bintray settings */
bintrayPackageLabels := Seq("nlp", "nlu",
  "natural-language-processing", "natural-language-understanding",
  "spark", "spark-ml", "pyspark", "machine-learning",
  "named-entity-recognition", "sentiment-analysis", "lemmatizer", "spell-checker",
  "tokenizer", "stemmer", "part-of-speech-tagger", "annotation-framework")

bintrayRepository := "spark-nlp"

bintrayOrganization:= Some("johnsnowlabs")

sonatypeProfileName := "com.johnsnowlabs"

publishTo := Some(
  if (isSnapshot.value)
    Opts.resolver.sonatypeSnapshots
  else
    Opts.resolver.sonatypeStaging
)

homepage:= Some(url("https://nlp.johnsnowlabs.com"))

scmInfo:= Some(
  ScmInfo(
    url("https://github.com/JohnSnowLabs/spark-nlp"),
    "scm:git@github.com:JohnSnowLabs/spark-nlp.git"
  )
)

developers in ThisBuild:= List(
  Developer(id="saifjsl", name="Saif Addin", email="saif@johnsnowlabs.com", url=url("https://github.com/saifjsl")),
  Developer(id="maziyarpanahi", name="Maziyar Panahi", email="maziyar@johnsnowlabs.com", url=url("https://github.com/maziyarpanahi")),
  Developer(id="albertoandreottiATgmail", name="Alberto Andreotti", email="alberto@pacific.ai", url=url("https://github.com/albertoandreottiATgmail")),
  Developer(id="danilojsl", name="Danilo Burbano", email="danilo@johnsnowlabs.com", url=url("https://github.com/danilojsl")),
  Developer(id="rohit13k", name="Rohit Kumar", email="rohit@johnsnowlabs.com", url=url("https://github.com/rohit13k")),
  Developer(id="aleksei-ai", name="Aleksei Alekseev", email="aleksei@pacific.ai", url=url("https://github.com/aleksei-ai")),
  Developer(id="showy", name="Eduardo Muñoz", email="eduardo@johnsnowlabs.com", url=url("https://github.com/showy")),
  Developer(id="C-K-Loan", name="Christian Kasim Loan", email="christian@johnsnowlabs.com", url=url("https://github.com/C-K-Loan")),
  Developer(id="wolliq", name="Stefano Lori", email="stefano@johnsnowlabs.com", url=url("https://github.com/wolliq")),
  Developer(id="vankov", name="Ivan Vankov", email="ivan@johnsnowlabs.com", url=url("https://github.com/vankov")),
  Developer(id="alinapetukhova", name="Alina Petukhova", email="alina@johnsnowlabs.com", url=url("https://github.com/alinapetukhova"))
)

lazy val analyticsDependencies = Seq(
  "org.apache.spark" %% "spark-core" % sparkVer % Provided,
  "org.apache.spark" %% "spark-mllib" % sparkVer % Provided
)

lazy val testDependencies = Seq(
  "org.scalatest" %% "scalatest" % scalaTestVersion % "test"
)

lazy val utilDependencies = Seq(
  typesafe,
  rocksdbjni,
  awsjavasdkbundle
    exclude("com.fasterxml.jackson.core", "jackson-annotations")
    exclude("com.fasterxml.jackson.core", "jackson-databind")
    exclude("com.fasterxml.jackson.core", "jackson-core")
    exclude("commons-configuration","commons-configuration"),
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
  if(is_gpu.equals("true"))
    Seq(tensorflowGPU)
  else
    Seq(tensorflowCPU)

lazy val mavenProps = settingKey[Unit]("workaround for Maven properties")

lazy val root = (project in file("."))
  .settings(
    crossScalaVersions := supportedScalaVersions,
    libraryDependencies ++=
      analyticsDependencies ++
        testDependencies ++
        utilDependencies ++
        tensorflowDependencies++
        typedDependencyParserDependencies,
    // TODO potentially improve this?
    mavenProps := {sys.props("javacpp.platform.extension") = if (is_gpu.equals("true")) "-gpu" else "" }
  )

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", "versions", "9", "module-info.class")  => MergeStrategy.discard
  case PathList("apache.commons.lang3", _ @ _*)  => MergeStrategy.discard
  case PathList("org.apache.hadoop", _ @ _*)  => MergeStrategy.first
  case PathList("com.amazonaws", _ @ _*)  => MergeStrategy.last
  case PathList("com.fasterxml.jackson") => MergeStrategy.first
  case PathList("META-INF", "io.netty.versions.properties")  => MergeStrategy.first
  case PathList("org", "tensorflow", _ @ _*)  => MergeStrategy.first
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}

/** Test tagging start */
// Command line fast:test
lazy val FastTest = config("fast") extend Test
// Command line slow:test
lazy val SlowTest = config("slow") extend Test

configs(FastTest, SlowTest)

parallelExecution in Test := false
logBuffered in Test := false
testOptions in Test := Seq(Tests.Argument("-l", "com.johnsnowlabs.tags.SlowTest")) // exclude

inConfig(FastTest)(Defaults.testTasks)
testOptions in FastTest := Seq(Tests.Argument("-l", "com.johnsnowlabs.tags.SlowTest")) // exclude
parallelExecution in FastTest := false

inConfig(SlowTest)(Defaults.testTasks)
testOptions in SlowTest := Seq(Tests.Argument("-n", "com.johnsnowlabs.tags.SlowTest")) // include
parallelExecution in SlowTest := false
/** Test tagging end */

/** Enable for debugging */
testOptions in Test += Tests.Argument("-oF")

/** Disables tests in assembly */
test in assembly := {}

/** Publish test artifact **/
publishArtifact in Test := true

/** Copies the assembled jar to the pyspark/lib dir **/
lazy val copyAssembledJar = taskKey[Unit]("Copy assembled jar to pyspark/lib")
lazy val copyAssembledJarForPyPi = taskKey[Unit]("Copy assembled jar to pyspark/sparknlp/lib")

copyAssembledJar := {
  val jarFilePath = (assemblyOutputPath in assembly).value
  val newJarFilePath = baseDirectory( _ / "python" / "lib" /  "sparknlp.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}

copyAssembledJarForPyPi := {
  val jarFilePath = (assemblyOutputPath in assembly).value
  val newJarFilePath = baseDirectory( _ / "python" / "sparknlp" / "lib"  / "sparknlp.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}
