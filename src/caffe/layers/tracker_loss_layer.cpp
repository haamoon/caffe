#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/sequence_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/tracking_layers.hpp"

namespace caffe {
  
  // Debug: Prob some of the outputs
  //const string debug_probs_arr[] = {"rO", "best_track"};
  //const vector<string> debug_probs(debug_probs_arr, debug_probs_arr+(sizeof(debug_probs_arr)/sizeof(debug_probs_arr[0])));
  //
  
  
  template <typename Dtype>
  void TrackerLossLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
    names->resize(1);
    (*names)[0] = "o_0";
  }
  
  template <typename Dtype>
  void TrackerLossLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
    names->resize(1);
    (*names)[0] = "o_" + this->int_to_str(this->T_);
  }
  
  template <typename Dtype>
  void TrackerLossLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
    const int object_num = this->layer_param_.tracker_loss_param().object_num();
    CHECK_GT(object_num, 0) << "object_num must be positive.";
    shapes->resize(1);
    //o_t is a 1 x N x object_num x 1 matrix
    (*shapes)[0].Clear();
    (*shapes)[0].add_dim(1);  // a single timestep
    (*shapes)[0].add_dim(this->N_);
    (*shapes)[0].add_dim(object_num);
    (*shapes)[0].add_dim(1);
  }
  
  template <typename Dtype>
  void TrackerLossLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
    names->resize(1);
    (*names)[0] = "loss";
    
    // Debug: Prob some of the outputs
    //names->resize(1 + debug_probs.size());
    //for(int i = 0; i < debug_probs.size(); i++) {
     // (*names)[i + 1] = ("out_" + debug_probs[i]);
    //}
    //
  }
  
  template <typename Dtype>
  void TrackerLossLayer<Dtype>::InputBlobNames(vector<string>* names) const {
    names->resize(3);
    (*names)[0] = "v";
    (*names)[1] = "gt_overlaps";
    (*names)[2] = "gt_num";
  }
  
  template <typename Dtype>
  void TrackerLossLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
    
    // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
    // use to save redundant code.
    LayerParameter slice_param;
    slice_param.set_type("Slice");
    slice_param.mutable_slice_param()->set_axis(0);
    
    CHECK_GE(net_param->input_size(), 2);
    CHECK_EQ(net_param->input(0).compare("v"), 0);
    CHECK_EQ(net_param->input(1).compare("gt_overlaps"), 0);
    
    // v_shape: T x N x num_track x num_seg
    const BlobShape v_shape = net_param->input_shape(0);
    
    // gt_overpas_shape: T x N x num_object x num_seg
    const BlobShape gt_overlaps_shape = net_param->input_shape(1);
    
    // split g_overlaps into gt1_overlpas and gt2_overlaps
    LayerParameter* gt_overlaps_split_param = net_param->add_layer();
    gt_overlaps_split_param->set_type("Split");
    gt_overlaps_split_param->set_name("gt_overlaps_split");
    gt_overlaps_split_param->add_bottom("gt_overlaps");
    gt_overlaps_split_param->add_top("gt1_overlaps");
    gt_overlaps_split_param->add_top("gt2_overlaps");
    
    //find the best tracks for each ground truth
    {      
      //1- Reshape gt1_overlaps from T x N x num_object x num_seg to T x (Nxnum_object) x 1 x num_seg
      // It is necessary to get true idxs which starts from 0..
      LayerParameter* r0_param = net_param->add_layer();
      r0_param->set_type("Reshape");
      r0_param->set_name("reshape0");
      r0_param->add_bottom("gt1_overlaps");
      r0_param->add_top("rgt1_overlaps");
      BlobShape* r0_top_blob_shape = r0_param->mutable_reshape_param()->mutable_shape();
      r0_top_blob_shape->add_dim(gt_overlaps_shape.dim(0));
      r0_top_blob_shape->add_dim(gt_overlaps_shape.dim(1) * gt_overlaps_shape.dim(2));
      r0_top_blob_shape->add_dim(1);
      r0_top_blob_shape->add_dim(gt_overlaps_shape.dim(3));
      
      // 2- compute best track for each segment (Input is a T x N x num_object x num_seg matrix) ==> (Output is a T x N x num_object x 1 matrix)
      LayerParameter* best_track_param = net_param->add_layer();
      best_track_param->set_type("Pooling");
      best_track_param->set_name("best_seg_match");
      best_track_param->mutable_pooling_param()->set_pool(PoolingParameter_PoolMethod_MAX);
      //best_track_param->mutable_pooling_param()->set_kernel_h(1);
      //best_track_param->mutable_pooling_param()->set_kernel_w(v_shape.dim(3));
      best_track_param->mutable_pooling_param()->set_global_pooling(true);
      best_track_param->add_bottom("rgt1_overlaps");
      best_track_param->add_top("tmp");
      best_track_param->add_top("rbest_track");
      
      
      LayerParameter* silent_tmp_param = net_param->add_layer();
      silent_tmp_param->set_type("Silence");
      silent_tmp_param->set_name("silent_tmp");
      silent_tmp_param->add_bottom("tmp");
      
      
      // rbest_track shape: T x (Nxnum_object) x 1 ==> best_track shape: T x N x num_object x 1
      LayerParameter* r1_param = net_param->add_layer();
      r1_param->set_type("Reshape");
      r1_param->set_name("reshape_best_track");
      r1_param->add_bottom("rbest_track");
      r1_param->add_top("best_track");
      BlobShape* r1_top_blob_shape = r1_param->mutable_reshape_param()->mutable_shape();
      r1_top_blob_shape->add_dim(gt_overlaps_shape.dim(0));
      r1_top_blob_shape->add_dim(gt_overlaps_shape.dim(1));
      r1_top_blob_shape->add_dim(gt_overlaps_shape.dim(2));
      r1_top_blob_shape->add_dim(1);
    }
    
    //
    vector<BlobShape> input_shapes;
    RecurrentInputShapes(&input_shapes);
    CHECK_EQ(1, input_shapes.size());
    net_param->add_input("o_0");
    net_param->add_input_shape()->CopyFrom(input_shapes[0]);
    
    LayerParameter* cont_slice_param = net_param->add_layer();
    cont_slice_param->CopyFrom(slice_param);
    cont_slice_param->set_name("cont_slice");
    cont_slice_param->add_bottom("cont");
    cont_slice_param->mutable_slice_param()->set_axis(0);
    
    LayerParameter* btrack_slice_param = net_param->add_layer();
    btrack_slice_param->CopyFrom(slice_param);
    btrack_slice_param->set_name("best_track_slice");
    btrack_slice_param->add_bottom("best_track");
    btrack_slice_param->mutable_slice_param()->set_axis(0);

    
    LayerParameter ob_concat_layer;
    ob_concat_layer.set_name("ob_concat");
    ob_concat_layer.set_type("Concat");
    ob_concat_layer.add_top("O");
    ob_concat_layer.mutable_concat_param()->set_axis(0);
    
    for (int t = 1; t <= this->T_; ++t) {
      string tm1s = this->int_to_str(t - 1);
      string ts = this->int_to_str(t);
      
      cont_slice_param->add_top("cont_" + ts);
      btrack_slice_param->add_top("bt_" + ts);
      
      // Add layers to compute
      //     new_o_{t} = bt_{t} * (1 - cont{t}) + o_{t-1} * cont{t}
      {
        LayerParameter* v_cont_param = net_param->add_layer();
        v_cont_param->set_type("Switch");
        v_cont_param->set_name("o_cont_" + tm1s);
        v_cont_param->add_bottom("bt_" + ts);
        v_cont_param->mutable_switch_param()->set_axis(2);
        v_cont_param->add_bottom("o_" + tm1s);
        v_cont_param->add_bottom("cont_" + ts);
        v_cont_param->add_top("new_o_" + ts);
      }
      
      // Add layers to splite new_o_{t} to o_{t}, and track_{t}
      {
        LayerParameter* o_split_param = net_param->add_layer();
        o_split_param->set_type("Split");
        o_split_param->set_name("o_split_" + ts);
        o_split_param->add_bottom("new_o_" + ts);
        o_split_param->add_top("o_" + ts);
        o_split_param->add_top("track_" + ts);
      }
      
      ob_concat_layer.add_bottom("track_" + ts);
    }  // for (int t = 1; t <= this->T_; ++t)
    net_param->add_layer()->CopyFrom(ob_concat_layer);
    
    //Reshape O from T x N x num_object x 1 ==> T x N x num_object
    {
      LayerParameter* r1_param = net_param->add_layer();
      r1_param->set_type("Reshape");
      r1_param->set_name("reshape1");
      r1_param->add_bottom("O");
      r1_param->add_top("rO");
      BlobShape* r1_top_blob_shape = r1_param->mutable_reshape_param()->mutable_shape();
      r1_top_blob_shape->add_dim(gt_overlaps_shape.dim(0));
      r1_top_blob_shape->add_dim(gt_overlaps_shape.dim(1));
      r1_top_blob_shape->add_dim(gt_overlaps_shape.dim(2));
    }
    
    //Add layer to computer gt' = HotMult(O, V)
    {
       LayerParameter* hotmult_param = net_param->add_layer();
       hotmult_param->set_type("HotMult");
       hotmult_param->set_name("hotmult");
       hotmult_param->mutable_hotmult_param()->set_mode("ROW");
       hotmult_param->add_bottom("rO");
       hotmult_param->add_bottom("v");
       hotmult_param->add_bottom("gt_num");
       hotmult_param->add_top("gt_bar");
    }
    
    //Add layer to compute loss ||gt' - gt||
    { 
      //Reshape from T x N x seg_num x gt_num to (TxNxseg_numxgt_num) x 1 x 1 x 1
      LayerParameter* reshape2_param = net_param->add_layer();
      reshape2_param->set_type("Reshape");
      reshape2_param->set_name("reshape2");
      reshape2_param->add_bottom("gt_bar");
      reshape2_param->add_top("rgt_bar");
      
      BlobShape* r2_top_blob_shape = reshape2_param->mutable_reshape_param()->mutable_shape();
      r2_top_blob_shape->add_dim(-1);
      r2_top_blob_shape->add_dim(1);
      r2_top_blob_shape->add_dim(1);
      r2_top_blob_shape->add_dim(1);
 
      //Reshape from T x N x seg_num x gt_num to (TxNxseg_numxgt_num) x 1 x 1 x 1
      LayerParameter* reshape3_param = net_param->add_layer();
      reshape3_param->set_type("Reshape");
      reshape3_param->set_name("reshape3");
      reshape3_param->add_bottom("gt2_overlaps");
      reshape3_param->add_top("rgt2_overlaps");
      
      BlobShape* r3_top_blob_shape = reshape3_param->mutable_reshape_param()->mutable_shape();
      r3_top_blob_shape->add_dim(-1);
      r3_top_blob_shape->add_dim(1);
      r3_top_blob_shape->add_dim(1);
      r3_top_blob_shape->add_dim(1);
      
      // Compute L2 loss
      LayerParameter* loss_param = net_param->add_layer();
      loss_param->set_type("EuclideanLoss");
      loss_param->set_name("loss");
      loss_param->add_bottom("rgt2_overlaps");
      loss_param->add_bottom("rgt_bar");
      loss_param->add_top("loss");
    }
    //
      
    // Debug: Prob some of the outputs
    //for(int i = 0; i < debug_probs.size(); i++) {
    //  LayerParameter* debug_prob_layer = net_param->add_layer();
    //  debug_prob_layer->set_type("Split");
    //  debug_prob_layer->set_name("debug_" + debug_probs[i]);
    //  debug_prob_layer->add_bottom(debug_probs[i]);
    //  debug_prob_layer->add_top("out_" + debug_probs[i]);
    //}
  }
  INSTANTIATE_CLASS(TrackerLossLayer);
  REGISTER_LAYER_CLASS(TrackerLoss);
}  // namespace caffe
