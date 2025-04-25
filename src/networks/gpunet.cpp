#include "booksim.hpp"
#include <vector>
#include <sstream>
#include <cmath>

#include "gpunet.hpp"
#include "misc_utils.hpp"

GPUNet::GPUNet( const Configuration& config, const string & name )
: Network ( config, name )
{
  _ComputeSize( config );
  _Alloc( );
  _BuildNet( config );
}

void GPUNet::_ComputeSize( const Configuration& config)
{
  _use_cpc = (config.GetInt( "use_cpc") == 1);
  _use_partition = (config.GetInt( "use_partition") == 1);
  _use_dsmem = (config.GetInt( "use_dsmem") == 1);

  _num_sms = config.GetInt( "sm" );
  _num_tpcs = config.GetInt( "tpc" );
  _num_cpcs = _use_cpc ? config.GetInt( "cpc" ) : 0;
  _num_gpcs = config.GetInt( "gpc" );
  _num_partitions = _use_partition ? config.GetInt( "partition" ) : 1;
  _num_l2groups = config.GetInt( "l2group" );
  _num_l2slices = config.GetInt( "l2slices" );

  _tpc_speedup = config.GetInt( "tpc_speedup" );
  _cpc_speedup = _use_cpc ? config.GetInt( "cpc_speedup" ) : _tpc_speedup;
  _gpc_speedup = config.GetInt( "gpc_speedup" );
  _inter_partition_speedup = config.GetInt( "interpartition_speedup" );
  _l2group_speedup = config.GetInt( "l2group_speedup" );

  // # routers used for a SM-to-L2 network
  _num_sm_to_l2 = _num_sms + _num_tpcs + _num_cpcs + _num_gpcs
                + _num_partitions + _num_l2groups + _num_l2slices;
  // # routers used for a SM-to-SM network if _use_dsmsm is set
  _num_sm_to_sm = _use_dsmem ? _num_tpcs + _num_cpcs + _num_gpcs : 0;  // crossbars

  _sms_per_tpc = _num_sms / _num_tpcs;
  if (_use_cpc) {
    _tpcs_per_cpc = _num_tpcs / _num_cpcs;
    _xpcs_per_gpc = _num_cpcs / _num_gpcs;
  } else
    _xpcs_per_gpc = _num_tpcs / _num_gpcs;
  _gpcs_per_partition = _num_gpcs / _num_partitions;
  _l2groups_per_partition = _num_l2groups / _num_partitions;
  _l2slices_per_l2group = _num_l2slices / _num_l2groups;

  // how to deal with dsmem inject/eject channels?
  _nodes = _num_sms + _num_l2slices; 

  _size = _num_sm_to_l2 + _num_sm_to_sm;

  _channels_sm_to_l2 = 2 * (_num_sms + _num_tpcs
           + _num_cpcs + _num_gpcs)                     // up/down across hierarchy
           + _num_partitions * (_num_partitions - 1)    // inter-partition
           + 2 * (_num_l2groups + _num_l2slices);

  _channels_sm_to_sm = _use_dsmem ? 
           2 * (_num_sms + _num_tpcs + _num_cpcs) : 0;

  _channels = _channels_sm_to_l2 + _channels_sm_to_sm;

  _l2slice_coords.resize(_num_l2slices);
  for ( int i = 0; i < _num_l2slices; ++i ) {
    // need to revisit
    _l2slice_coords[i] = make_pair(i % 8, i / 8);
  }

  _n = config.GetInt( "n" );
  _k = config.GetInt( "k" );
  
  gN = _n;
  gK = _k;
}

void GPUNet::_BuildNet( const Configuration& config)
{
  ostringstream name;
  int id, latency;
  int layer = 0;

  //
  // Allocate SM-to-L2 Network's Routers
  //
  // SM routers (router_sm_sm)
  for ( int sm = 0; sm < _num_sms; ++sm) {
    name.str("");
    name << "router_sm_" << sm;
    id = _GetRouterIndex( layer, sm, false );
    _routers[id] = Router::NewRouter( config, this, name.str( ),
             id, _use_dsmem ? 3 : 2, _use_dsmem ? 3 : 2);
    _timed_modules.push_back(_routers[id]);
  }
  layer++;

  // TPC routers (router_g_c_t)
  for ( int t = 0; t < _num_tpcs; ++t ) {
    int g = t / (_num_tpcs / _num_gpcs);
    name.str("");
    name << "router_" << g << "_";
    if (_use_cpc) {
      int c = t / (_num_tpcs / _num_cpcs);
      name << c << "_";
    }
    name << "t";
    id = _GetRouterIndex( layer, t, false );
    _routers[id] = Router::NewRouter( config, this, name.str( ),
             id, _sms_per_tpc + 1, _sms_per_tpc + 1);
    _timed_modules.push_back(_routers[id]);
  }
  layer++;

  // CPC routers (router_g_c)
  if (_use_cpc) {
    for ( int c = 0; c < _num_cpcs; ++c ) {
      int g = c / (_num_cpcs / _num_gpcs);
      name.str("");
      name << "router_" << g << "_" << c;
      id = _GetRouterIndex( layer, c, false );
      _routers[id] = Router::NewRouter( config, this, name.str( ),
               id, _tpcs_per_cpc + 1, _tpcs_per_cpc + 1);
      _timed_modules.push_back(_routers[id]);
    }
    layer++;
  }

  // GPC routers (router_g)
  for ( int g = 0; g < _num_gpcs; ++g ) {
    name.str("");
    name << "router_" << g;
    id = _GetRouterIndex( layer, g, false );
    _routers[id] = Router::NewRouter( config, this, name.str( ),
             id, _xpcs_per_gpc + 1, _xpcs_per_gpc + 1);
    _timed_modules.push_back(_routers[id]);
  }
  layer++;

  // Crossbars for each partition (router_crossbar_p)
  for ( int p = 0; p < _num_partitions; ++p) {
    name.str("");
    name << "router_crossbar_" << p;
    id = _GetRouterIndex( layer, p, false );
    int inter_partition = (_use_partition && _num_partitions > 1) ? _num_partitions -1 : 0;
    _routers[id] = Router::NewRouter( config, this, name.str( ), id,
             _gpcs_per_partition + _l2groups_per_partition + inter_partition,
             _gpcs_per_partition + _l2groups_per_partition + inter_partition);
    _timed_modules.push_back(_routers[id]);
  }
  layer++;

  // L2 groups routers (router_l2_l2g)
  for ( int l2g = 0; l2g < _num_l2groups; ++l2g ) {
    name.str("");
    name << "router_l2_" << l2g;
    id = _GetRouterIndex( layer, l2g, false );
    _routers[id] = Router::NewRouter( config, this, name.str( ),
             id, _l2slices_per_l2group + 1, _l2slices_per_l2group + 1);
    _timed_modules.push_back(_routers[id]);
  }
  layer++;

  // L2 slice routers (router_l2_l2g_l2s)
  for ( int l2 = 0; l2 < _num_l2slices; ++l2 ) {
    int l2g = l2 / _l2slices_per_l2group;
    name.str("");
    name << "router_l2_" << l2g << "_" << l2;
    id = _GetRouterIndex( layer, l2, false );
    _routers[id] = Router::NewRouter( config, this, name.str( ),
             id, 2, 2);
    _timed_modules.push_back(_routers[id]);
  }

  layer = 1;
  //
  // Allocate SM-to-SM Network's Routers
  //
  if (_use_dsmem) {
    // TPC crossbars (router_crossbar_g_c_t)
    for ( int t = 0; t < _num_tpcs; ++t ) {
      int g = t / (_num_tpcs / _num_gpcs);
      name.str("");
      name << "router_crossbar_" << g << "_";
      if (_use_cpc) {
        int c = t / (_num_tpcs / _num_cpcs);
        name << c << "_";
      }
      name << "t";
      id = _GetRouterIndex( layer, t, true );
      _routers[id] = Router::NewRouter( config, this, name.str( ),
              id, _sms_per_tpc + 1, _sms_per_tpc + 1);
      _timed_modules.push_back(_routers[id]);
    }
    layer++;

    // CPC crossbars (router_crossbar_g_c)
    if (_use_cpc) {
      for ( int c = 0; c < _num_cpcs; ++c ) {
        int g = c / (_num_cpcs / _num_gpcs);
        name.str("");
        name << "router_crossbar_" << g << "_" << c;
        id = _GetRouterIndex( layer, c, true );
        _routers[id] = Router::NewRouter( config, this, name.str( ),
                id, _tpcs_per_cpc + 1, _tpcs_per_cpc + 1);
        _timed_modules.push_back(_routers[id]);
      }
      layer++;
    }

    // GPC crossbars (router_crossbar_g)
    for ( int g = 0; g < _num_gpcs; ++g ) {
      name.str("");
      name << "router_crossbar_" << g;
      id = _GetRouterIndex( layer, g, true );
      _routers[id] = Router::NewRouter( config, this, name.str( ),
              id, _xpcs_per_gpc, _xpcs_per_gpc);
      _timed_modules.push_back(_routers[id]);
    }
  }
  layer = 0;

  //
  // Connect Channels to Routers
  //
  // Injection & Ejection Channels
  for ( int node = 0; node < _nodes; ++node ) {
    int sm_router_id = (node < _num_sms) ? _GetRouterIndex( 0, node, false ) : -1;
    int l2slice_router_id = (node >= _num_sms) ? _GetRouterIndex( 6, node - _num_sms, false ) : -1;

    if( sm_router_id != -1 ) {
      _routers[sm_router_id]->AddInputChannel( _inject[node], _inject_cred[node] );
      _routers[sm_router_id]->AddOutputChannel( _eject[node], _eject_cred[node] );
    } else if ( l2slice_router_id != -1 ) {
      _routers[l2slice_router_id]->AddInputChannel( _inject[node], _inject_cred[node] );
      _routers[l2slice_router_id]->AddOutputChannel( _eject[node], _eject_cred[node] );
    }

    latency = _WireLatency( layer, false );

    _inject[node]->SetLatency( 1 );
    _inject_cred[node]->SetLatency( 1 );
    _inject[node]->SetBandwidth( 1 );
    _inject_cred[node]->SetBandwidth( 1 );
    
    _eject[node]->SetLatency( 1 );
    _eject_cred[node]->SetLatency( 1 );
    _eject[node]->SetBandwidth( 1 );
    _eject_cred[node]->SetBandwidth( 1 );
  }
  
  // SM-to-L2 Network + SM-to-SM Network
  int c_id = 0;
  for ( int t = 0; t < _num_tpcs; ++t ) {
    for ( int s = 0; s < _sms_per_tpc; ++s ) {

      int sm_id = t * _sms_per_tpc + s;
      int lower = _GetRouterIndex( layer, sm_id, false );
      int higher = _GetRouterIndex( layer + 1, t, false );

      latency = _WireLatency( layer, false );

      // SM-to-TPC in the SM-to-L2 Network
      // upward direction
      _routers[lower]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
      _routers[higher]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );

      _chan[c_id]->SetLatency( latency );
      _chan_cred[c_id]->SetLatency( latency );
      _chan[c_id]->SetBandwidth( 1 );
      _chan_cred[c_id]->SetBandwidth( 1 );

      c_id++;

      // downward direction
      _routers[lower]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
      _routers[higher]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
      
      _chan[c_id]->SetLatency( latency );
      _chan_cred[c_id]->SetLatency( latency );
      _chan[c_id]->SetBandwidth( 1 );
      _chan_cred[c_id]->SetBandwidth( 1 );

      c_id++;
      
      // SM-to-TPC in the SM-to-SM Network
      if (_use_dsmem) {
        
        int lower_d = _GetRouterIndex( layer, sm_id, true);
        int higher_d = _GetRouterIndex( layer + 1, t, true );

        latency = _WireLatency( layer, true );

        // upward
        _routers[lower_d]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
        _routers[higher_d]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );

        _chan[c_id]->SetLatency( latency );
        _chan_cred[c_id]->SetLatency( latency );
        _chan[c_id]->SetBandwidth( 1 );
        _chan_cred[c_id]->SetBandwidth( 1 );

        c_id++;

        // downward
        _routers[lower_d]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
        _routers[higher_d]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
        
        _chan[c_id]->SetLatency( latency );
        _chan_cred[c_id]->SetLatency( latency );
        _chan[c_id]->SetBandwidth( 1 );
        _chan_cred[c_id]->SetBandwidth( 1 );

        c_id++;
      }    
    }
  }
  layer++;

  // TPC-to-CPC 
  if (_use_cpc) {
    for ( int c = 0; c < _num_cpcs; ++c ) {
      for ( int t = 0; t < _tpcs_per_cpc; ++t ) {

        int tpc_id = c * _tpcs_per_cpc + t;
        int lower = _GetRouterIndex( layer, tpc_id, false );
        int higher = _GetRouterIndex( layer + 1 , c, false);

        latency = _WireLatency( layer, false );

        // TPC-to-CPC Network in the SM-to-L2 Network
        // upward direction
        _routers[lower]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
        _routers[higher]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );

        _chan[c_id]->SetLatency( latency );
        _chan_cred[c_id]->SetLatency( latency );
        _chan[c_id]->SetBandwidth( _tpc_speedup );
        _chan_cred[c_id]->SetBandwidth( _tpc_speedup );

        c_id++;

        // downward direction
        _routers[lower]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
        _routers[higher]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
        
        _chan[c_id]->SetLatency( latency );
        _chan_cred[c_id]->SetLatency( latency );
        _chan[c_id]->SetBandwidth( _tpc_speedup );
        _chan_cred[c_id]->SetBandwidth( _tpc_speedup );

        c_id++;

        // TPC-to-CPC in the SM-to-SM Network
        if (_use_dsmem) {
          int lower_d = _GetRouterIndex( layer, tpc_id, true);
          int higher_d = _GetRouterIndex( layer + 1, c, true );

          latency = _WireLatency( layer, true );

          // upward
          _routers[lower_d]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
          _routers[higher_d]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );

          _chan[c_id]->SetLatency( latency );
          _chan_cred[c_id]->SetLatency( latency );
          _chan[c_id]->SetBandwidth( _tpc_speedup );
          _chan_cred[c_id]->SetBandwidth( _tpc_speedup );

          c_id++;

          // downward
          _routers[lower_d]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
          _routers[higher_d]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
          
          _chan[c_id]->SetLatency( latency );
          _chan_cred[c_id]->SetLatency( latency );
          _chan[c_id]->SetBandwidth( _tpc_speedup );
          _chan_cred[c_id]->SetBandwidth( _tpc_speedup );

          c_id++;
        }
      }
    }
    layer++;
  }
  
  // CPC-to-GPC or TPC-to-GPC
  for ( int g = 0; g < _num_gpcs; ++g ) {
    for ( int x = 0; x < _xpcs_per_gpc; ++x ) {

      int xpc_id = g * _xpcs_per_gpc + x;
      int lower = _GetRouterIndex( layer, xpc_id, false );
      int higher = _GetRouterIndex( layer + 1, g, false);

      latency = _WireLatency( layer, false );

      // CPC/TPC-to-GPC in the SM-to-L2 Network
      // upward direction
      _routers[lower]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
      _routers[higher]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
      
      _chan[c_id]->SetLatency( latency );
      _chan_cred[c_id]->SetLatency( latency );
      _chan[c_id]->SetBandwidth( _cpc_speedup );
      _chan_cred[c_id]->SetBandwidth( _cpc_speedup );

      c_id++;

      // downward direction
      _routers[lower]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
      _routers[higher]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
      
      _chan[c_id]->SetLatency( latency );
      _chan_cred[c_id]->SetLatency( latency );
      _chan[c_id]->SetBandwidth( _cpc_speedup );
      _chan_cred[c_id]->SetBandwidth( _cpc_speedup );

      c_id++;
      // CPC/TPC-to-GPC in the SM-to-SM Network
      if (_use_dsmem) {
        int lower_d = _GetRouterIndex( layer, xpc_id, true);
        int higher_d = _GetRouterIndex( layer + 1, g, true );

        latency = _WireLatency( layer, true );

        // upward
        _routers[lower_d]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
        _routers[higher_d]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );

        _chan[c_id]->SetLatency( latency );
        _chan_cred[c_id]->SetLatency( latency );
        _chan[c_id]->SetBandwidth( _cpc_speedup );
        _chan_cred[c_id]->SetBandwidth( _cpc_speedup );

        c_id++;

        // downward
        _routers[lower_d]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
        _routers[higher_d]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
        
        _chan[c_id]->SetLatency( latency );
        _chan_cred[c_id]->SetLatency( latency );
        _chan[c_id]->SetBandwidth( _cpc_speedup );
        _chan_cred[c_id]->SetBandwidth( _cpc_speedup );

        c_id++;
      }
    }
  }
  layer++;

  // GPC-to-Crossbar
  for ( int p = 0; p < _num_partitions; ++p ) {
    for ( int g = 0; g < _gpcs_per_partition; ++g ) {

      int gpc_id = p * _gpcs_per_partition + g;
      int lower = _GetRouterIndex( layer, gpc_id, false );
      int higher = _GetRouterIndex( layer + 1, p, false);

      latency = _WireLatency( layer, false );

      // upward direction
      _routers[lower]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
      _routers[higher]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
      
      _chan[c_id]->SetLatency( latency );
      _chan_cred[c_id]->SetLatency( latency );
      _chan[c_id]->SetBandwidth( _gpc_speedup );
      _chan_cred[c_id]->SetBandwidth( _gpc_speedup );

      c_id++;

      // downward direction
      _routers[lower]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
      _routers[higher]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
      
      _chan[c_id]->SetLatency( latency );
      _chan_cred[c_id]->SetLatency( latency );
      _chan[c_id]->SetBandwidth( _gpc_speedup );
      _chan_cred[c_id]->SetBandwidth( _gpc_speedup );

      c_id++;
    }
  }
  layer++;

  // Crossbar-to-Crossbar
  if (_use_partition && _num_partitions > 1) {
    for ( int p1 = 0; p1 < _num_partitions; ++p1 ) {
      for ( int p2 = p1 + 1; p2 < _num_partitions; ++p2 ) {
        int p1_id = _GetRouterIndex( layer, p1, false );
        int p2_id = _GetRouterIndex( layer, p2, false );

        latency = _WireLatency( layer, false );

        // p1 -> p2
        _routers[p1_id]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
        _routers[p2_id]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
        
        _chan[c_id]->SetLatency( latency );
        _chan_cred[c_id]->SetLatency( latency );
        _chan[c_id]->SetBandwidth( _inter_partition_speedup );
        _chan_cred[c_id]->SetBandwidth( _inter_partition_speedup );

        c_id++;

        // p2 -> p1
        _routers[p1_id]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
        _routers[p2_id]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
        
        _chan[c_id]->SetLatency( latency );
        _chan_cred[c_id]->SetLatency( latency );
        _chan[c_id]->SetBandwidth( _inter_partition_speedup );
        _chan_cred[c_id]->SetBandwidth( _inter_partition_speedup );

        c_id++;
      }
    }
  }

  // Crossbar-to-L2Group
  for ( int p = 0; p < _num_partitions; ++p ) {
    for ( int l2g = 0; l2g < _l2groups_per_partition; ++l2g ) {

      int l2group_id = p * _l2groups_per_partition + l2g;
      int lower = _GetRouterIndex( layer, p, false );
      int higher = _GetRouterIndex( layer + 1, l2group_id, false);

      latency = _WireLatency( layer, false );

      // upward direction
      _routers[lower]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
      _routers[higher]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
      
      _chan[c_id]->SetLatency( latency );
      _chan_cred[c_id]->SetLatency( latency );
      _chan[c_id]->SetBandwidth( _l2group_speedup );
      _chan_cred[c_id]->SetBandwidth( _l2group_speedup );

      c_id++;

      // downward direction
      _routers[lower]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
      _routers[higher]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
      
      _chan[c_id]->SetLatency( latency );
      _chan_cred[c_id]->SetLatency( latency );
      _chan[c_id]->SetBandwidth( _l2group_speedup );
      _chan_cred[c_id]->SetBandwidth( _l2group_speedup );

      c_id++;
    }
  }
  layer++;


  // L2Group-to-L2Slice
  for ( int l2g = 0; l2g < _num_l2groups; ++l2g ) {
    for ( int l2 = 0; l2 < _l2slices_per_l2group; ++l2 ) {

      int l2slice_id = l2g * _l2slices_per_l2group + l2;
      int lower = _GetRouterIndex( layer, l2g, false );
      int higher = _GetRouterIndex( layer + 1, l2slice_id, false);

      latency = _WireLatency( layer, false );

      // upward direction
      _routers[lower]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
      _routers[higher]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
      
      _chan[c_id]->SetLatency( latency );
      _chan_cred[c_id]->SetLatency( latency );
      _chan[c_id]->SetBandwidth( 1 );
      _chan_cred[c_id]->SetBandwidth( 1 );

      c_id++;

      // downward direction
      _routers[lower]->AddInputChannel( _chan[c_id], _chan_cred[c_id] );
      _routers[higher]->AddOutputChannel( _chan[c_id], _chan_cred[c_id] );
      
      _chan[c_id]->SetLatency( latency );
      _chan_cred[c_id]->SetLatency( latency );
      _chan[c_id]->SetBandwidth( 1 );
      _chan_cred[c_id]->SetBandwidth( 1 );

      c_id++;
    }
  }
}

int GPUNet::_GetRouterIndex( int layer, int id, bool dsmem )
{
  int index = 0;
  // SM-to-L2 Network
  if (!dsmem) {
    if ( layer == 0 ) return id;          // SM
    index += _num_sms;
    if ( layer == 1 ) return index + id;  // TPC
    index += _num_tpcs;
    if ( layer == 2 ) return index + id;  // CPC
    index += _num_cpcs;
    if ( layer == 3 ) return index + id;  // GPC
    index += _num_gpcs;
    if ( layer == 4 ) return index + id;  // Crossbars
    index += _num_partitions;
    if ( layer == 5 ) return index + id;  // L2 Groups
    index += _num_l2groups;
    if ( layer == 6 ) return index + id;  // L2 Slices
  } else {
  // SM-to-SM network
    if ( layer == 0 ) return id;          // SM
    index += _num_sm_to_l2;
    if ( layer == 1 ) return index + id;  // Crossbars in TPCs
    index += _num_tpcs;
    if ( layer == 2 ) return index + id;  // Crossbars in CPCs
    index += _num_cpcs;
    if ( layer == 3 ) return index + id;  // Crossbars in GPCs
  }

  return -1;
}

int GPUNet::_WireLatency(int layer, bool dsmem)
{
  // latency calculation according to the hierarchy
  int base_latency = 1;
  if ( layer == 0 ) return base_latency;
  base_latency += 2;
  if ( layer == 1 ) return base_latency;
  base_latency += 4;
  if ( layer == 2 ) return base_latency;
  base_latency += 6;
  if ( layer == 3 ) return base_latency;
  base_latency += 8;
  if ( layer == 4 ) return base_latency;
  base_latency += 10;
  if ( layer == 5 ) return base_latency;
  base_latency += _FloorplanLatency( 0, 0, 0, 0);
  if ( layer == 6 ) return base_latency;  // L2 Slices

  return -1;
}

int GPUNet::_FloorplanLatency( int src_x, int src_y, int dst_x, int dst_y)
{
  return abs( src_x - dst_x ) + abs( src_y - dst_y );
}

