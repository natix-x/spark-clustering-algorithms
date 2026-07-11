name := "spark-clustering"
version := "0.1"

scalaVersion := "2.12.15"

val sparkVersion = "3.3.2"

resolvers += "SparkPackages" at "https://repos.spark-packages.org/"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"   % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql"    % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib"  % sparkVersion % "provided",
  "graphframes" % "graphframes" % "0.8.2-spark3.2-s_2.12",
  "org.apache.commons" % "commons-math3" % "3.6.1",
  "org.log4s" %% "log4s" % "1.10.0",
  "org.apache.logging.log4j" % "log4j-slf4j2-impl" % "2.20.0" % "provided"
)

assembly / mainClass := Some("clustering.benchmark.BenchmarkRunner")
assembly / assemblyJarName := "spark-clustering-benchmark.jar"

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x if x.endsWith("module-info.class") => MergeStrategy.discard
  case x => MergeStrategy.first
}

exportJars := true