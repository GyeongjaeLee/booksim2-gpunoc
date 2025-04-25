
#ifndef _GPUNET_HPP_
#define _GPUNET_HPP_
#include <cassert>
#include <vector>
#include "network.hpp"

class GPUNet : public Network {

  int _n;
  int _k;

  bool _use_partition;
  bool _use_cpc;
  bool _use_dsmem;

  int _num_sms;
  int _num_tpcs;
  int _num_cpcs;
  int _num_gpcs;
  int _num_partitions;
  int _num_l2groups;
  int _num_l2slices;

  int _tpc_speedup;
  int _cpc_speedup;
  int _gpc_speedup;
  int _inter_partition_speedup;
  int _l2group_speedup;

  int _sms_per_tpc;
  int _tpcs_per_cpc;
  // if _use_cpc=1, cpcs/gpc, otherwise, tpcs/gpc
  int _xpcs_per_gpc;
  int _gpcs_per_partition;
  int _l2groups_per_partition;
  int _l2slices_per_l2group;



  int _num_sm_to_l2;
  int _num_sm_to_sm;

  int _channels_sm_to_l2;
  int _channels_sm_to_sm;



  vector<pair<int, int> > _l2slice_coords;

  void _ComputeSize( const Configuration& config );
  void _BuildNet( const Configuration& config );

  int _GetRouterIndex( int layer, int id, bool dsmem );

  int _WireLatency( int layer, bool dsmem );
  int _FloorplanLatency( int src_x, int src_y, int dst_x, int dst_y );
  int _ChannelBandwidth( int layer, bool dsmem );

public:

  GPUNet( const Configuration& config, const string & name );
  static void RegisterRoutingFunctions() ;

};

#endif
