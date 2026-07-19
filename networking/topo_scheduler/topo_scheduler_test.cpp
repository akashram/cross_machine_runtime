// topo_scheduler_test.cpp — models an 8-GPU node: two 4-GPU NVSwitch
// groups (300 GB/s within a group) bridged by one slow cross-group link
// (25 GB/s, standing in for a PCIe/NIC hop). Workload has four
// high-volume "tensor-parallel duo" task pairs and three low-volume
// "pipeline-adjacent" pairs linking the duos together. A topology-aware
// placement should keep every high-volume duo within one NVSwitch group.

#include "topo_scheduler.h"

#include <cstdio>

int main() {
  topo::TopologyGraph g;
  int groupA[4], groupB[4];
  for (int i = 0; i < 4; ++i) groupA[i] = g.addNode("gpuA" + std::to_string(i));
  for (int i = 0; i < 4; ++i) groupB[i] = g.addNode("gpuB" + std::to_string(i));

  for (int i = 0; i < 4; ++i)
    for (int j = i + 1; j < 4; ++j) {
      g.addLink(groupA[i], groupA[j], 300.0);
      g.addLink(groupB[i], groupB[j], 300.0);
    }
  g.addLink(groupA[0], groupB[0], 25.0); // the one slow cross-group bridge

  topo::Workload w;
  w.numTasks = 8;
  w.commVolumeGb.assign(8, std::vector<double>(8, 0.0));
  auto setVol = [&](int i, int j, double v) { w.commVolumeGb[i][j] = w.commVolumeGb[j][i] = v; };
  setVol(0, 1, 100); setVol(2, 3, 100); setVol(4, 5, 100); setVol(6, 7, 100); // TP duos
  setVol(1, 2, 1); setVol(3, 4, 1); setVol(5, 6, 1); // pipeline links between duos

  std::vector<int> placement = topo::place(g, w);

  bool inSameGroup[4] = {false, false, false, false};
  int duoTasks[4][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}};
  for (int d = 0; d < 4; ++d) {
    int nodeA = placement[duoTasks[d][0]];
    int nodeB = placement[duoTasks[d][1]];
    bool aInGroupA = nodeA < 4, bInGroupA = nodeB < 4;
    inSameGroup[d] = (aInGroupA == bInGroupA);
    std::printf("duo %d (tasks %d,%d) -> nodes %s,%s: %s\n", d, duoTasks[d][0], duoTasks[d][1],
                g.nodeNames[static_cast<size_t>(nodeA)].c_str(),
                g.nodeNames[static_cast<size_t>(nodeB)].c_str(),
                inSameGroup[d] ? "same NVSwitch group" : "SPLIT ACROSS GROUPS");
  }

  bool ok = inSameGroup[0] && inSameGroup[1] && inSameGroup[2] && inSameGroup[3];
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
