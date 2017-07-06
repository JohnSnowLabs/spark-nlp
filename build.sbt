
val scalaLangVersion = "2.11.11"
val sparkVersion = "2.1.1"
val scalaTestVersion = "3.0.0"
val snowballVersion = "1.0"
val opennlpVersion = "1.6.0"
val scalanlpNERVersion = "2015.2.19"

lazy val analyticsDependencies = Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.tartarus" % "snowball" % snowballVersion,
  "org.scalanlp" %% "epic-ner-en-conll" % scalanlpNERVersion,
  "org.apache.opennlp" % "opennlp-tools" % opennlpVersion
)

lazy val testDependencies = Seq(
  "org.scalatest" %% "scalatest" % scalaTestVersion,
  "org.scalactic" %% "scalactic" % scalaTestVersion,
  "com.storm-enroute" %% "scalameter" % "0.8.2"
)

lazy val utilDependencies = Seq(
  "com.typesafe" % "config" % "1.3.0"
)

resolvers += "Collide repo" at "http://mvn.collide.info/content/repositories/releases/"
resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/releases"

testFrameworks += new TestFramework("org.scalameter.ScalaMeterFramework")

parallelExecution in Test := false
logBuffered in Test := false

scalacOptions ++= Seq(
  "-feature",
  "-language:implicitConversions"
)

lazy val root = (project in file("."))
  .settings(
    name := "sparknlp",
    version := "1.0.0",
    organization := "com.jsl.nlp",
    scalaVersion := scalaLangVersion,
    libraryDependencies ++=
      analyticsDependencies ++
      testDependencies ++
      utilDependencies
  )
