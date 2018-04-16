val sparkVer = "2.3.0"
val scalaVer = "2.11.12"
val scalaTestVersion = "3.0.0"

/** Package attributes */
name := "spark-nlp"

organization := "com.johnsnowlabs.nlp"

version := "1.5.1"

scalaVersion := scalaVer

sparkVersion := sparkVer

/** Spark-Package attributes */
spName := "JohnSnowLabs/spark-nlp"

sparkComponents ++= Seq("mllib")

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

spIncludeMaven := false

spAppendScalaVersion := false

resolvers += "Maven Central" at "http://central.maven.org/maven2/"

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
  Developer(id="aleksei-ai", name="Aleksei Alekseev", email="aleksei@pacific.ai", url=url("https://github.com/aleksei-ai"))
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
  "org.slf4j" % "slf4j-api" % "1.7.25",
  "org.apache.commons" % "commons-compress" % "1.15",
  "org.tensorflow" % "tensorflow" % "1.5.0",
  "com.amazonaws" % "aws-java-sdk-s3" % "1.11.313"
)

lazy val root = (project in file("."))
  .settings(
    libraryDependencies ++=
      analyticsDependencies ++
        testDependencies ++
        utilDependencies
  )

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

copyAssembledJar := {
  val jarFilePath = (assemblyOutputPath in assembly).value
  val newJarFilePath = baseDirectory( _ / "python" / "lib" /  "sparknlp.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}
