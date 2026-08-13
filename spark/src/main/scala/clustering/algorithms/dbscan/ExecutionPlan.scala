package clustering.algorithms.dbscan

private[dbscan] sealed trait ExecutionPlan
private[dbscan] object ExecutionPlan {
  final case class Distributed(rowsPerBlock: Int) extends ExecutionPlan
  final case class DriverLocal(reason: String) extends ExecutionPlan
}
