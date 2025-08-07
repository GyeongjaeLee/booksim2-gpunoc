#include "booksim.hpp"
#include <vector>
#include <sstream>
#include <cmath>

#include "gpunet.hpp"
#include "misc_utils.hpp"
#include "globals.hpp"

int gX; // # of partition crossbars
vector<int> gU; // units per layer

GPUNet::GPUNet( const Configuration& config, const string & name )
: Network ( config, name )
{
  _ComputeSize(config);
  _Alloc();
  _BuildNet(config);
}

void GPUNet::_ComputeSize( const Configuration& config )
{
  // Number of layers
  _l = config.GetInt("l");
  _partition = (config.GetInt("partition") == 1);
  
  // Nodes (SM and L2 slices)
  _nodes_sm = config.GetInt("sm");
  _nodes_l2slice = config.GetInt("l2slice");
  
  _nodes = _nodes_sm + _nodes_l2slice;

  _ratio = config.GetIntArray("units");
  if (_ratio.empty() || (_ratio.size() < size_t(_l))) {
    _ratio.resize(_l, 1);
  }
  
  _total_units.resize(_l);
  for (int l = 0; l < _l; ++l) {
    if (l == 0)
      _total_units[l] = _nodes_sm / _ratio[l];
    else
      _total_units[l] = _total_units[l - 1] / _ratio[l];
  }

  _offsets.resize(_l, 0);
  for (int l = 1; l < _l; ++l) {
    _offsets[l] = _total_units[l - 1] + _offsets[l - 1];
  }

  // Routers for SM-to-L2 Request and Reply Network
  _size = 0;
  for (int l = 0; l < _l; l++) {
    _size += 2 * _total_units[l];
  }

  // Channels for SM-to-L2 Network
  _channels = 0;
  for (int l = 0; l < _l; l++) {
    if (l < _l - 1) {
      _channels += 2 * _total_units[l];
    } else {
      // Fully-connected partitioned crossbars
      _p = _partition ? _total_units[l] : 1;
      _channels += 2 * _p * (_p - 1);
    }
  }

  _l2slice_p = _nodes_l2slice / _p; // L2 slices per partition

  _speedups = config.GetIntArray("speedups");
  if (_speedups.empty() || (_speedups.size() < size_t(_l + 1))) {
    _speedups.resize(_l + 1, 1);
  }

  _inter_partition_speedup = config.GetInt("inter_partition_speedup");

#ifdef GPUNET_DEBUG
  cout << "GPUNet Configuration:" << endl;
  cout << "  l: " << _l << endl;
  cout << "  nodes_sm: " << _nodes_sm << endl;
  cout << "  nodes_l2slice: " << _nodes_l2slice << endl;
  cout << "  ratio: ";
  for (const auto& r : _ratio) {
    cout << " " << r;
  }
  cout << endl;
  cout << "  total_units: ";
  for (const auto& u : _total_units) {
    cout << " " << u;
  }
  cout << endl;
  cout << "  offsets: ";
  for (const auto& o : _offsets) {
    cout << " " << o;
  }
  cout << endl;
  cout << "  size: " << _size << endl;
  cout << "  channels: " << _channels << endl;
  cout << "  l2slice_p: " << _l2slice_p << endl;
  cout << "  speedups: ";
  for (const auto& s : _speedups) {
    cout << " " << s;
  }
  cout << endl;
  cout << "  inter_partition_speedup: " << _inter_partition_speedup << endl;
#endif

  gN = _l;
  gX = _p;
  gU = _ratio;
}

void GPUNet::_BuildNet(const Configuration& config)
{
  ostringstream name;
  int c, id;

  // STEP 1: Create all routers first
  for (int l = 0; l < _l; ++l) {
    for (int addr = 0; addr < _total_units[l]; ++addr) {
      id = _offsets[l] + addr;
      
      int bottom_ports = (l < _l - 1) ? _ratio[l] : (_ratio[l] + (_p - 1));
      int top_ports = (l < _l - 1) ? 1 : (_l2slice_p + (_p - 1));

      name.str("");
      name << "router_" << "request" << "_" << l << "_" << addr;
      _routers[id] = Router::NewRouter(config, this, name.str(), id, bottom_ports, top_ports);
      _timed_modules.push_back(_routers[id]);

      name.str("");
      name << "router_" << "reply" << "_" << l << "_" << addr;
      _routers[id + _size / 2] = Router::NewRouter(config, this, name.str(), id + _size / 2, top_ports, bottom_ports);
      _timed_modules.push_back(_routers[id + _size / 2]);
    }
  }
  
  // STEP 2: Connect SM->TPC (injection) and TPC->SM (ejection) first
#ifdef GPUNET_DEBUG
  cout << "Connecting SM nodes..." << endl;
#endif

  for (int addr = 0; addr < _total_units[0]; ++addr) {
    for (int port = 0; port < _ratio[0]; ++port) {
      id = _offsets[0] + addr;
      c = addr * _ratio[0] + port;  // SM node index

      // Request network: SM -> TPC (injection)
      _routers[id]->AddInputChannel(_inject[c], _inject_cred[c]);
      
      // Reply network: TPC -> SM (ejection)
      _routers[id + _size / 2]->AddOutputChannel(_eject[c], _eject_cred[c]);
    }
  }
  
  // STEP 3: Connect L2->Crossbar (injection) and Crossbar->L2 (ejection)
#ifdef GPUNET_DEBUG
  cout << "Connecting L2 nodes..." << endl;
#endif

  for (int addr = 0; addr < _total_units[_l - 1]; ++addr) {
    for (int port = 0; port < _l2slice_p; ++port) {
      id = _offsets[_l - 1] + addr;
      c = _nodes_sm + addr * _l2slice_p + port;  // L2 node index
      
      // Request network: Crossbar -> L2 (ejection)
      _routers[id]->AddOutputChannel(_eject[c], _eject_cred[c]);
      
      // Reply network: L2 -> Crossbar (injection)
      _routers[id + _size / 2]->AddInputChannel(_inject[c], _inject_cred[c]);
    }
  }
  
  // STEP 4: Connect internal network channels

#ifdef GPUNET_DEBUG
  cout << "Connecting internal channels of request network..." << endl;
#endif
  
  // 4.1: Connect Request Network internal channels
  for (int l = 0; l < _l; ++l) {
    for (int addr = 0; addr < _total_units[l]; ++addr) {
      id = _offsets[l] + addr;
      
      // Connect bottom channels (from lower layer)
      if (l > 0) {
        for (int port = 0; port < _ratio[l]; ++port) {
          c = _offsets[l - 1] + addr * _ratio[l] + port;
          _routers[id]->AddInputChannel(_chan[c], _chan_cred[c]);
        }
      }

      // Connect top channels (to higher layer)
      if (l < _l - 1) {
        c = _offsets[l] + addr;
        _routers[id]->AddOutputChannel(_chan[c], _chan_cred[c]);
      }

      // Connect inter-partition channels for the last layer
      if (l == _l - 1) {
        for (int port = 0; port < (_p - 1); ++port) {
          int src_partition = port;
          if (src_partition >= addr) {
            src_partition++;
          }

          int src_outport = addr;
          if (src_outport > src_partition) {
            src_outport--;
          }
          
          // For each partition router, output and input channels
          // are connected in sequential port order.
          // Output channel
          c = _offsets[l] + addr * (_p - 1) + port;
          _routers[id]->AddOutputChannel(_chan[c], _chan_cred[c]);

#ifdef GPUNET_DEBUG
          cout << "Connecting inter-partition channel " << c
               << " as an output chnanel of partition " << addr
               << " through outport " << port << endl;
#endif
          
          // Input channel
          c = _offsets[l] + src_partition * (_p - 1) + src_outport;
          _routers[id]->AddInputChannel(_chan[c], _chan_cred[c]);

#ifdef GPUNET_DEBUG
          cout << "Connecting inter-partition channel " << c
               << " as an input channel from partition " << src_partition
               << " using outport " << src_outport
               << " to partition " << addr
               << " through inport " << port << endl;
#endif

        }
      }
    }
  }
  
#ifdef GPUNET_DEBUG
  cout << "Connecting internal channels of reply network..." << endl;
#endif

  // 4.2: Connect Reply Network internal channels
  for (int l = 0; l < _l; ++l) {
    for (int addr = 0; addr < _total_units[l]; ++addr) {
      id = _offsets[l] + addr + _size / 2;
      
      // Connect bottom channels (to lower layer)
      if (l > 0) {
        for (int port = 0; port < _ratio[l]; ++port) {
          c = _offsets[l - 1] + addr * _ratio[l] + port + _channels / 2;
          _routers[id]->AddOutputChannel(_chan[c], _chan_cred[c]);
        }
      }

      // Connect top channels (from higher layer)
      if (l < _l - 1) {
        c = _offsets[l] + addr + _channels / 2;
        _routers[id]->AddInputChannel(_chan[c], _chan_cred[c]);
      }
      
      // Connect inter-partition channels for the last layer
      if (l == _l - 1) {
        for (int port = 0; port < (_p - 1); ++port) {
          int src_partition = port;
          if (src_partition >= addr) {
            src_partition++;
          }

          int src_outport = addr;
          if (src_outport > src_partition) {
            src_outport--;
          }

          // For each partition router, output and input channels
          // are connected in sequential port order.
          // Output channel
          c = _offsets[l] + addr * (_p - 1) + port + _channels / 2;
          _routers[id]->AddOutputChannel(_chan[c], _chan_cred[c]);
          
#ifdef GPUNET_DEBUG
          cout << "Connecting inter-partition channel " << c
               << " as an output chnanel of partition " << addr
               << " through outport " << port << endl;
#endif
          // Input channel
          c = _offsets[l] + src_partition * (_p - 1) + src_outport + _channels / 2;
          _routers[id]->AddInputChannel(_chan[c], _chan_cred[c]);

#ifdef GPUNET_DEBUG
          cout << "Connecting inter-partition channel " << c
               << " as an input channel from partition " << src_partition
               << " using outport " << src_outport
               << " to partition " << addr
               << " through inport " << port << endl;
#endif

        }
      }
    }
  }

  _SetupChannels();
}

// Set up all channel properties
void GPUNet::_SetupChannels()
{
  cout << "Setting up channel properties..." << endl;
  
  // Injection and Ejection channels
  for (int i = 0; i < _nodes_sm; i++) {
    _SetChannelProperties(_inject[i], _inject_cred[i], 0);
    _SetChannelProperties(_eject[i], _eject_cred[i], 0);
  }

  for (int i = _nodes_sm; i < _nodes; i++) {
    _SetChannelProperties(_inject[i], _inject_cred[i], _l);
    _SetChannelProperties(_eject[i], _eject_cred[i], _l);
  }
  
  for (int l = 1; l < _l; l++) {
    int start = _offsets[l - 1];
    int end = _offsets[l];
    
    // TPC <-> CPC, CPC <-> GPC, GPC <-> Crossbar channels
    for (int c = start; c < end; c++) {
      _SetChannelProperties(_chan[c], _chan_cred[c], l);
      _SetChannelProperties(_chan[c + _channels / 2], _chan_cred[c + _channels / 2], l);
    }
  }
  
  // interpartition channels for the last layer
  if (_partition) {
    int p_start = _offsets[_l - 1];
    int p_end = _offsets[_l - 1] + _p * (_p - 1);
    for (int c = p_start; c < p_end; c++) {
      _SetChannelProperties(_chan[c], _chan_cred[c], _l - 1, true);
      _SetChannelProperties(_chan[c + _channels / 2], _chan_cred[c + _channels / 2], _l - 1, true);
    }
  }
  
  
  cout << "All channel properties set" << endl;
}

// Set channel latency and bandwidth based on layer properties
void GPUNet::_SetChannelProperties(FlitChannel* channel, CreditChannel* credit_channel, int layer, bool is_inter_partition)
{
  int latency = _GetWireLatency(layer, is_inter_partition);
  int bandwidth = _GetChannelBandwidth(layer, is_inter_partition);
  
  channel->SetLatency(latency);
  channel->SetBandwidth(bandwidth);
  credit_channel->SetLatency(latency);
  credit_channel->SetBandwidth(bandwidth);
}

int GPUNet::_GetWireLatency(int layer, bool is_inter_partition) const
{
  // Base latency increases with layer depth
  int latency = 1 + layer;
  
  // Add additional latency for inter-partition connections
  if (is_inter_partition) {
    latency += 1;
  }
  
  return latency;
}

int GPUNet::_GetChannelBandwidth(int layer, bool is_inter_partition) const
{
  return is_inter_partition ? _inter_partition_speedup : _speedups[layer];
}

int GPUNet::_FloorplanLatency(int src_x, int src_y, int dst_x, int dst_y) const
{
  return abs(src_x - dst_x) + abs(src_y - dst_y);
}


void GPUNet::RegisterRoutingFunctions()
{
  gRoutingFunctionMap["hierarchical_gpunet"] = &hierarchical_gpunet;
  // gRoutingFunctionMap["direct_fullyconnected"] = &direct_fullyconnected;

}

void hierarchical_gpunet(const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject)
{
  int vcBegin = 0, vcEnd = gNumVCs - 1;
  if (f->type == Flit::READ_REQUEST || f->type == Flit::READ_REPLY) {
    vcBegin = 0;
    vcEnd = gNumVCs / 2 - 1;
  } else if (f->type == Flit::WRITE_REQUEST || f->type == Flit::WRITE_REPLY) {
    vcBegin = gNumVCs / 2;
    vcEnd = gNumVCs - 1;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;
  
  if (inject) {
    // injection can use all VCs
    outputs->AddRange(-1, vcBegin, vcEnd);
    return;
  }

  int src = f->src;
  int dest = f->dest;
  int hops = f->hops;

  // cout << "router: " << r->GetID() 
  //      << ", time: " << GetSimTime() << endl;

  // cout << "Routing flit: " << f->id
  //      <<  " src: " << src 
  //      << ", dest: " << dest 
  //      << ", hops: " << f->hops << endl;

  // # of SMs and L2 Slices per partition
  int sm = gX;
  for (int i = 0; i < gN; ++i) {
    sm *= gU[i];
  }
  int sm_p = sm / gX; // SMs per partition
  int l2slice = gNodes - sm;
  int l2slice_p = l2slice / gX; // L2 slices per partition

  assert((src < sm && dest >= sm) || (src >= sm && dest < sm));
  bool is_request = dest > src;

  int src_partition = is_request ? (src / sm_p) : ((src - sm) / l2slice_p);
  int dest_partition = is_request ? ((dest - sm) / l2slice_p) : (dest / sm_p);
  bool is_remote = (dest_partition != src_partition);

  // if remote access is required, add one additional hop,
  // in the case of fully-connected partitioned network
  int total_hops = is_remote ? (gN + 1) : gN;
  int cur_layer = is_request ? hops : (total_hops - hops - 1);

  if (is_request) {
    // request network
    if (cur_layer < gN - 1) {
      out_port = 0;
    } else {
      // partition layer
      assert(r->NumOutputs() == l2slice_p + (gX - 1));

      if (is_remote && (cur_layer == total_hops - 2)) {
        // inter-partition communication
        int dest_port = (dest_partition > src_partition) ? (dest_partition - 1) : dest_partition;
        out_port = l2slice_p + dest_port;
      } else {
        // intra-partition communication or already crossed inter-partition
        out_port = (dest - sm) % l2slice_p;
      }
    }
  } else {
    // reply network
    if (cur_layer < gN - 1) {
      int sm_group = 1;
      for (int i = 0; i < cur_layer; ++i) {
        sm_group *= gU[i];
      }
      out_port = (dest % (sm_group * gU[cur_layer])) / sm_group;
    } else {
      // partition layer
      assert(r->NumInputs() == l2slice_p + (gX - 1));

      if (is_remote && (cur_layer == total_hops - 1)) {
        // inter-partition communication
        int dest_port = (dest_partition > src_partition) ? (dest_partition - 1) : dest_partition; 
        out_port = gU[gN - 1] + dest_port;
      } else {
        // intra-partition communication or already crossed inter-partition
        out_port = (dest % sm_p) / (sm_p / gU[gN - 1]);
      }
    }
  }

  if (f->watch) {
    *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
               << "Adding VC range ["
               << vcBegin << ","
               << vcEnd << "]"
               << " at output port " << out_port
               << " for flit " << f->id
               << " (input port " << in_channel
               << ", destination " << f->dest << ")"
               << "." << endl;
  }

  outputs->Clear( );

  outputs->AddRange( out_port, vcBegin, vcEnd );
}