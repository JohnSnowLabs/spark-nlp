import sbtassembly.MergeStrategy

val sparkVer = "2.4.0"
val scalaVer = "2.11.12"
val scalaTestVersion = "3.0.0"

/** Package attributes */
name := "spark-nlp"

organization := "com.johnsnowlabs.nlp"

version := "1.8.0"

scalaVersion in ThisBuild := scalaVer

sparkVersion in ThisBuild := sparkVer

/** Spark-Package attributes */
spName in ThisBuild := "JohnSnowLabs/spark-nlp"

sparkComponents in ThisBuild ++= Seq("mllib")

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

spIncludeMaven := false

spAppendScalaVersion := false

resolvers in ThisBuild += "Maven Central" at "http://central.maven.org/maven2/"

assemblyOption in assembly := (assemblyOption in assembly).value.copy(
  includeScala = false
)

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

ivyScala := ivyScala.value map {
  _.copy(overrideScalaVersion = true)
}

/** Bintray settings */
bintrayPackageLabels := Seq("nlp", "nlu",
  "natural-language-processing", "natural-language-understanding",
  "spark", "spark-ml", "pyspark", "machine-learning",
  "named-entity-recognition", "sentiment-analysis", "lemmatizer", "spell-checker",
  "tokenizer", "stemmer", "part-of-speech-tagger", "annotation-framework")

bintrayRepository := "spark-nlp"

bintrayOrganization := Some("johnsnowlabs")

sonatypeProfileName := "com.johnsnowlabs"

publishTo := Some(
  if (isSnapshot.value)
    Opts.resolver.sonatypeSnapshots
  else
    Opts.resolver.sonatypeStaging
)

homepage := Some(url("https://nlp.johnsnowlabs.com"))

scmInfo := Some(
  ScmInfo(
    url("https://github.com/JohnSnowLabs/spark-nlp"),
    "scm:git@github.com:JohnSnowLabs/spark-nlp.git"
  )
)

developers := List(
  Developer(id="saifjsl", name="Saif Addin", email="saif@johnsnowlabs.com", url=url("https://github.com/saifjsl")),
  Developer(id="showy", name="Eduardo Muñoz", email="eduardo@johnsnowlabs.com", url=url("https://github.com/showy")),
  Developer(id="aleksei-ai", name="Aleksei Alekseev", email="aleksei@pacific.ai", url=url("https://github.com/aleksei-ai")),
  Developer(id="albertoandreottiATgmail", name="Alberto Andreotti", email="alberto@pacific.ai", url=url("https://github.com/albertoandreottiATgmail")),
  Developer(id="danilojsl", name="Danilo Burbano", email="danilo@johnsnowlabs.com", url=url("https://github.com/danilojsl"))
)


lazy val ocrDependencies = Seq(
  "net.sourceforge.tess4j" % "tess4j" % "4.2.1"
    exclude("org.slf4j", "slf4j-log4j12")
    exclude("org.apache.logging", "log4j"),
  "org.apache.pdfbox" % "pdfbox" % "2.0.13",
  "org.apache.pdfbox" % "jbig2-imageio" % "3.0.2"
)

lazy val analyticsDependencies = Seq(
  "org.apache.spark" %% "spark-core" % sparkVer % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVer % "provided"
)

lazy val testDependencies = Seq(
  "org.scalatest" %% "scalatest" % scalaTestVersion % "test"
)

lazy val utilDependencies = Seq(
  "com.typesafe" % "config" % "1.3.0",
  "org.rocksdb" % "rocksdbjni" % "5.1.4",
  "com.amazonaws" % "aws-java-sdk" % "1.7.4"
    exclude("commons-codec", "commons-codec")
    exclude("com.fasterxml.jackson.core", "jackson-core")
    exclude("com.fasterxml.jackson.core", "jackson-annotations")
    exclude("com.fasterxml.jackson.core", "jackson-databind")
    exclude("com.fasterxml.jackson.dataformat", "jackson-dataformat-smile")
    exclude("com.fasterxml.jackson.datatype", "jackson-datatype-joda"),
  "org.tensorflow" % "tensorflow" % "1.12.0",
  "com.github.universal-automata" % "liblevenshtein" % "3.0.0"
  exclude("com.google.guava", "guava"),
  "com.navigamez" % "greex" % "1.0"

  /** Enable the following for tensorflow GPU support */
  //"org.tensorflow" % "libtensorflow" % "1.12.0",
  //"org.tensorflow" % "libtensorflow_jni_gpu" % "1.12.0",
)

lazy val typedDependencyParserDependencies = Seq(
  "net.sf.trove4j" % "trove4j" % "3.0.3",
  "junit" % "junit" % "4.10" % Test
)

lazy val root = (project in file("."))
  .settings(
    libraryDependencies ++=
      analyticsDependencies ++
        testDependencies ++
        utilDependencies ++
        typedDependencyParserDependencies
  )

val ocrMergeRules: String => MergeStrategy  = {

  case "versionchanges.txt" => MergeStrategy.discard
  case "StaticLoggerBinder" => MergeStrategy.discard
  case PathList("META-INF", fileName)
    if List("NOTICE", "MANIFEST.MF", "DEPENDENCIES", "INDEX.LIST").contains(fileName) || fileName.endsWith(".txt")
        => MergeStrategy.discard
  case PathList("META-INF", "services", _ @ _*)  => MergeStrategy.first
  case PathList("META-INF", xs @ _*)  => MergeStrategy.first
  case PathList("org", "apache", _ @ _*)  => MergeStrategy.first
  case PathList("apache", "commons", "logging", "impl",  xs @ _*)  => MergeStrategy.discard
  case _ => MergeStrategy.deduplicate
}

assemblyMergeStrategy in assembly := {
  case PathList("com.fasterxml.jackson") => MergeStrategy.first
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}

lazy val ocr = (project in file("ocr"))
  .settings(
    name := "spark-nlp-ocr",
    version := "1.8.0",
    libraryDependencies ++= ocrDependencies ++
      analyticsDependencies ++
      testDependencies,
    assemblyMergeStrategy in assembly := ocrMergeRules
  )
  .dependsOn(root % "test")

parallelExecution in Test := false

logBuffered in Test := false

scalacOptions ++= Seq(
  "-feature",
  "-language:implicitConversions"
)

/** Enable for debugging */
testOptions in Test += Tests.Argument("-oF")

/** Disables tests in assembly */
test in assembly := {}

/** Publish test artificat **/
publishArtifact in Test := true

/** Copies the assembled jar to the pyspark/lib dir **/
lazy val copyAssembledJar = taskKey[Unit]("Copy assembled jar to pyspark/lib")
lazy val copyAssembledOcrJar = taskKey[Unit]("Copy assembled jar to pyspark/lib")
lazy val copyAssembledJarForPyPi = taskKey[Unit]("Copy assembled jar to pyspark/sparknlp/lib")

copyAssembledJar := {
  val jarFilePath = (assemblyOutputPath in assembly).value
  val newJarFilePath = baseDirectory( _ / "python" / "lib" /  "sparknlp.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}

copyAssembledOcrJar := {
  val jarFilePath = (assemblyOutputPath in assembly in "ocr").value
  val newJarFilePath = baseDirectory( _ / "python" / "lib" /  "sparknlp-ocr.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}

copyAssembledJarForPyPi := {
  val jarFilePath = (assemblyOutputPath in assembly).value
  val newJarFilePath = baseDirectory( _ / "python" / "sparknlp" / "lib"  / "sparknlp.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}