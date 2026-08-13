package clustering.algorithms.kmedoids.components

/** Replacing the medoid in `slot` with the point `candidate`, and what it does to the objective.
 *
 *  `costDelta >= 0` means "no improvement"; [[SwapMove.None]] is the neutral element of
 *  [[SwapMove.preferred]], so a parallel scan can reduce over slices that found nothing.
 */
private[kmedoids] final case class SwapMove(costDelta: Double, slot: Int, candidate: Int)

private[kmedoids] object SwapMove {

  val None: SwapMove = SwapMove(0.0, -1, -1)

  /** Lower Δ wins; ties go to the lower slot, then the lower candidate index. One rule for the
   *  whole ladder, so the rungs cannot disagree on an exact tie (see `docs/kmedoids_docs.md`). */
  def preferred(a: SwapMove, b: SwapMove): SwapMove =
    if (b.slot < 0) a
    else if (a.slot < 0) b
    else if (a.costDelta != b.costDelta) if (a.costDelta < b.costDelta) a else b
    else if (a.slot != b.slot) if (a.slot < b.slot) a else b
    else if (a.candidate <= b.candidate) a else b

  def isImprovement(move: SwapMove): Boolean = move.slot >= 0 && move.costDelta < 0.0
}
