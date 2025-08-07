#ifndef _GPUNET_HPP_
#define _GPUNET_HPP_
#include <cassert>
#include <vector>

#include "network.hpp"
#include "routefunc.hpp"

class GPUNet : public Network {
  
  // # of layer that requests traverse to get partition crossbars
  // ex) (_l == 3): SM->TPC->GPC->Crossbar, (_l == 4): SM->TPC->CPC->GPC->Crossbar
  int _l;

  int _nodes_sm;
  int _nodes_l2slice;
  int _l2slice_p;
  
  // # of lower-level units connected to a single higher-level module.
  // l =      0,             1,      ...,     _l-1  
  // ex) [SM per TPCs, TPCs per XPC, ..., GPCs per Crossbars]
  vector<int> _ratio;
  // ex) [TPCs, ..., GPCs, Crossbars]
  vector<int> _total_units;
  // _offsets for Router and Channel id of each layer
  vector<int> _offsets;
  // l =     0,    ...,   _l-2,     _l-1                 _l
  // ex) [SM->TPC, ..., CPC->GPC, GPC->Crossbar, Crossbar->L2Slice]
  // speedup for SM->TPC should always be 1 (channel width, not bandwidth?)
  vector<int> _speedups;
    
  // A100, H100 supports partitioned GPUNet, not supported in V100
  bool _partition;
  // # of partitions, _p = 1 for non-partitioned GPUNet
  int _p;
  int _inter_partition_speedup;

  

  vector<pair<int, int> > _l2slice_coords;

  void _ComputeSize(const Configuration& config);
  void _BuildNet(const Configuration& config);

  // Set latency and bandwidth for a channel based on its layer
  void _SetupChannels();
  void _SetChannelProperties(FlitChannel* channel, CreditChannel* credit_channel,
                   int layer, bool is_inter_partition = false);
  int _GetWireLatency(int layer, bool is_inter_partition = false) const;
  int _GetChannelBandwidth(int layer, bool is_inter_partition = false) const;
  
  int _WireLatency(int l) const;
  int _FloorplanLatency(int src_x, int src_y, int dst_x, int dst_y) const;

public:

  GPUNet( const Configuration& config, const string & name );
  static void RegisterRoutingFunctions();
};

void hierarchical_gpunet( const Router *r, const Flit *f, int in_channel,
                   OutputSet *outputs, bool inject );

#endif
