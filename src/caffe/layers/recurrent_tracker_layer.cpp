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
  //const string debug_probs_arr[] = {"mx_1", "xmx_1", "m_1", "c_1", "c_2", "h_1", "h_2", "w_0", "w_1", "hm1_0", "hm1_1", "m_2"};
  //const vector<string> debug_probs(debug_probs_arr, debug_probs_arr+(sizeof(debug_probs_arr)/sizeof(debug_probs_arr[0])));
  //
  
  // Debug: Prob some of the outputs
  //template <typename Dtype>
  //inline int RecurrentTrackerLayer<Dtype>::ExactNumTopBlobs() const {
  //  return 2 + debug_probs.size(); 
  //}
  
  template <typename Dtype>
  void RecurrentTrackerLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
    names->resize(3);
    (*names)[0] = "X";
    (*names)[1] = "V";
    (*names)[2] = "cont";
  }
  
  template <typename Dtype>
  void RecurrentTrackerLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
    names->resize(1);
    (*names)[0] = "Y";
    //(*names)[1] = "h_" + this->int_to_str(this->T_);
  }
  
  template <typename Dtype>
  void RecurrentTrackerLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
    
    const int max_nseg = this->layer_param_.recurrent_tracker_param().max_nseg();
    CHECK_GT(max_nseg, 0) << "max_nseg must be positive.";

    const int feature_dim = this->layer_param_.recurrent_tracker_param().feature_dim();
    CHECK_GT(feature_dim, 0) << "feature_dim must be positive.";
    
    const int max_ntrack = this->layer_param_.recurrent_tracker_param().max_ntrack();
    CHECK_GT(max_ntrack, 0) << "max_ntrack must be positive";
    
    shapes->resize(3);
    //X is a T x N x max_nseg x feature_dim matrix
    (*shapes)[0].Clear();
    (*shapes)[0].add_dim(this->T_);
    (*shapes)[0].add_dim(this->N_);
    (*shapes)[0].add_dim(max_nseg);
    (*shapes)[0].add_dim(feature_dim);
    
    //V is a T x N x max_nseg x max_ntrack matrix
    (*shapes)[1].Clear();
    (*shapes)[1].add_dim(this->T_);
    (*shapes)[1].add_dim(this->N_);
    (*shapes)[1].add_dim(max_nseg);
    (*shapes)[1].add_dim(max_ntrack);

    //cont is a T x N matrix
    (*shapes)[2].Clear();
    (*shapes)[2].add_dim(this->T_);
    (*shapes)[2].add_dim(this->N_);
  }
  
  template <typename Dtype>
  void RecurrentTrackerLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
    names->resize(1);
    (*names)[0] = "Y";
    // Debug: Prob some of the outputs
    //names->resize(2 + debug_probs.size());
    //for(int i = 0; i < debug_probs.size(); i++) {
    //  (*names)[i + 2] = ("out_" + debug_probs[i]);
    //}
    //
  }
  
  template <typename Dtype>
  void RecurrentTrackerLayer<Dtype>::InputBlobNames(vector<string>* names) const {
    names->resize(3);
    (*names)[0] = "X";
    (*names)[1] = "V";
    (*names)[2] = "cont";
  }
  
  template <typename Dtype>
  void RecurrentTrackerLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
    const int max_ntrack = this->layer_param_.recurrent_tracker_param().max_ntrack();
    
    // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
    // use to save redundant code.
    LayerParameter matmult_param;
    matmult_param.set_type("MatMult");
    
    
    LayerParameter sum_param;
    sum_param.set_type("Eltwise");
    sum_param.mutable_eltwise_param()->set_operation(
                                                     EltwiseParameter_EltwiseOp_SUM);
    
    LayerParameter slice_param;
    slice_param.set_type("Slice");
    slice_param.mutable_slice_param()->set_axis(0);
    
    LayerParameter scalar_param;
    scalar_param.set_type("Scalar");
    scalar_param.mutable_scalar_param()->set_axis(0);
    
    
    //TODO: currently feature_dim_ can not be set because this
    //fucntion is defined as constant. So, I set it mannually
    //in RecurrentInputShapes proto.
    //initialize feature_dim_ before calling RecurrentInputShapes
    //x input is a T_ x N_ x max_nseg x feature_dim_ array
    CHECK_GE(net_param->input_size(), 1);
    CHECK_EQ(net_param->input(0).compare("x"), 0);
    const BlobShape input_blob_shape = net_param->input_shape(0);
    
    //Check max_nseg >= max_ntrack
    CHECK_GE(input_blob_shape.dim(2), max_ntrack) << "Number of segments " << input_blob_shape.dim(2) << " should be greater than or equal to the number of track " << max_ntrack;
    
    vector<BlobShape> input_shapes;
    RecurrentInputShapes(&input_shapes);
    CHECK_EQ(2, input_shapes.size());
    net_param->add_input("c_0");
    net_param->add_input_shape()->CopyFrom(input_shapes[0]);
    net_param->add_input("h_0");
    net_param->add_input_shape()->CopyFrom(input_shapes[1]);
    
    LayerParameter* cont_slice_param = net_param->add_layer();
    cont_slice_param->CopyFrom(slice_param);
    cont_slice_param->set_name("cont_slice");
    cont_slice_param->add_bottom("cont");
    cont_slice_param->mutable_slice_param()->set_axis(0);
    
    LayerParameter* x_slice_param = net_param->add_layer();
    x_slice_param->CopyFrom(slice_param);
    x_slice_param->set_name("x_slice");
    x_slice_param->add_bottom("x");
    
    LayerParameter* overlaps_slice_param = net_param->add_layer();
    overlaps_slice_param->CopyFrom(slice_param);
    overlaps_slice_param->set_name("overlaps_slice");
    overlaps_slice_param->add_bottom("overlaps");
    
    LayerParameter* nseg_slice_param = net_param->add_layer();
    nseg_slice_param->CopyFrom(slice_param);
    nseg_slice_param->set_name("seg_num_slice");
    nseg_slice_param->add_bottom("seg_num");
    
    
    LayerParameter output_concat_layer;
    output_concat_layer.set_name("v_concat");
    output_concat_layer.set_type("Concat");
    output_concat_layer.add_top("v");
    output_concat_layer.mutable_concat_param()->set_axis(0);
    
    
    
    LayerParameter vtilde_concat_layer;
    vtilde_concat_layer.set_name("vtilde_concat");
    vtilde_concat_layer.set_type("Concat");
    vtilde_concat_layer.add_top("vtilde");
    vtilde_concat_layer.mutable_concat_param()->set_axis(0);
    
    
    for (int t = 1; t <= this->T_; ++t) {
      string tm1s = this->int_to_str(t - 1);
      string ts = this->int_to_str(t);
      
      cont_slice_param->add_top("cont_" + ts);
      x_slice_param->add_top("x_" + ts);
      overlaps_slice_param->add_top("overlaps_" + ts);
      nseg_slice_param->add_top("nseg_" + ts);
      
      
      // Add layers to split overlaps_{t} to overlapsi_{t} i = 1,2
      {
        LayerParameter* overlaps_split_param = net_param->add_layer();
        overlaps_split_param->set_type("Split");
        overlaps_split_param->set_name("overlaps_split_" + ts);
        overlaps_split_param->add_bottom("overlaps_" + ts);
        overlaps_split_param->add_top("overlaps1_" + ts);
        overlaps_split_param->add_top("overlaps2_" + ts);
      }
      
      // Add layers to split x_{t} to xi_{t} i = 1,2,3,4
      {
        LayerParameter* x_split_param = net_param->add_layer();
        x_split_param->set_type("Split");
        x_split_param->set_name("x_split_" + ts);
        x_split_param->add_bottom("x_" + ts);
        x_split_param->add_top("x1_" + ts);
        x_split_param->add_top("x2_" + ts);
        x_split_param->add_top("x3_" + ts);
        x_split_param->add_top("x4_" + ts);
      }
      
      // Add layers to split cont_{t} to conti_{t} i = 1,2,3
      {
        LayerParameter* v_split_param = net_param->add_layer();
        v_split_param->set_type("Split");
        v_split_param->set_name("cont_split_" + ts);
        v_split_param->add_bottom("cont_" + ts);
        v_split_param->add_top("cont1_" + ts);
        v_split_param->add_top("cont2_" + ts);
        v_split_param->add_top("cont3_" + ts);
        v_split_param->add_top("cont4_" + ts);
      }
      
      // Add layers to split h_{t-1} to h1_{t-1} h2_{t-1}
      {
        LayerParameter* h_split_param = net_param->add_layer();
        h_split_param->set_type("Split");
        h_split_param->set_name("h_split_" + tm1s);
        h_split_param->add_bottom("h_" + tm1s);
        h_split_param->add_top("h1_" + tm1s);
        h_split_param->add_top("h2_" + tm1s);
      }
      
      
      // Add layers to split c_{t} to c1_{t} c2_{t}
      {
        LayerParameter* c_split_param = net_param->add_layer();
        c_split_param->set_type("Split");
        c_split_param->set_name("c_split_" + tm1s);
        c_split_param->add_bottom("c_" + tm1s);
        c_split_param->add_top("c1_" + tm1s);
        c_split_param->add_top("c2_" + tm1s);
      }
            
      // Add layers to compute
      //     hm1_{t-1} := (h_{t-1} + \lambda I)^{-1}
      {
        LayerParameter* hm1_param = net_param->add_layer();
        hm1_param->set_type("MatInv");
        hm1_param->mutable_matinv_param()->set_lambda(this->layer_param_.recurrent_tracker_param().lambda());
        hm1_param->set_name("hm1_" + tm1s);
        hm1_param->add_bottom("h1_" + tm1s);
        hm1_param->add_top("hm1_" + tm1s);
      }
      
      // Add layers to compute
      //     w_{t-1} := c_{t-1} hm1_{t-1}
      {
        LayerParameter* w_param = net_param->add_layer();
        w_param->CopyFrom(matmult_param);
        w_param->set_name("w_" + tm1s);
        w_param->add_bottom("c1_" + tm1s);
        w_param->add_bottom("hm1_" + tm1s);
        w_param->add_top("w_" + tm1s);
      }
      
      // Add layer to compute
      //     v'_{t} = w_{t-1} X_{t}^\top
      {
        LayerParameter* v_param = net_param->add_layer();
        v_param->CopyFrom(matmult_param);
        v_param->set_name("v_" + ts);
        v_param->add_bottom("w_" + tm1s);
        v_param->add_bottom("x_" + ts);
        v_param->mutable_matmult_param()->set_transpose_b(true);
        v_param->add_top("v_" + ts);
      }
      
      // Add layers to compute
      //     v_cont_{t} = v'_{t} * cont{t} + slice(overlaps_{t} * (1 - cont{t}))
      {
        //If the second output is empty slice layer will fail
        if(input_blob_shape.dim(2) - max_ntrack > 0) {
          LayerParameter* s_overlaps_param = net_param->add_layer();
          s_overlaps_param->set_type("Slice");
          s_overlaps_param->set_name("slice_overlaps_" + ts);
          s_overlaps_param->mutable_slice_param()->set_axis(2);
          s_overlaps_param->mutable_slice_param()->add_slice_point(max_ntrack);
          s_overlaps_param->add_bottom("overlaps1_" + ts);
          s_overlaps_param->add_top("s_overlaps_" + ts);
          s_overlaps_param->add_top("tmp_" + ts);
          
          LayerParameter* silent_tmp_param = net_param->add_layer();
          silent_tmp_param->set_type("Silence");
          silent_tmp_param->set_name("silent_tmp_" + ts);
          silent_tmp_param->add_bottom("tmp_" + ts);
        }
        //slice is not necessary
        else {
          LayerParameter* s_overlaps_param = net_param->add_layer();
          s_overlaps_param->set_type("Split");
          s_overlaps_param->set_name("rename_overlaps_" + ts);
          s_overlaps_param->add_bottom("overlaps1_" + ts);
          s_overlaps_param->add_top("s_overlaps_" + ts);
        }
        
        LayerParameter* v_cont_param = net_param->add_layer();
        v_cont_param->set_type("Switch");
        v_cont_param->set_name("v_cont_" + ts);
        v_cont_param->mutable_switch_param()->set_axis(2);
        v_cont_param->add_bottom("s_overlaps_" + ts);
        v_cont_param->add_bottom("v_" + ts);
        v_cont_param->add_bottom("cont1_" + ts);
        v_cont_param->add_top("v_cont_" + ts);
      }
      
      
      // Add layers to copy v_cont_{t} 3 times
      //     v1_{t}, v2_{t}, v3_{t}, v4_{t}
      //{
        LayerParameter* v_split_param = net_param->add_layer();
        v_split_param->set_type("Split");
        v_split_param->set_name("v_split_" + ts);
        v_split_param->add_bottom("v_cont_" + ts);
        // for computing m_{t}
        v_split_param->add_top("v1_" + ts);
        // for computing C_{t}
        //v_split_param->add_top("v2_" + ts);
        // for computing output V
        v_split_param->add_top("v3_" + ts);
        // for computing vtilde
        v_split_param->add_top("v4_" + ts);
      //}
      
      // Add layers to calculate v_tilde_{t}
      {
        if(this->layer_param_.recurrent_tracker_param().use_softmax()) {
          //TODO: handle the case where some of the tracks are not matched (zero row in v)
          //NOT_IMPLEMENTED;
          LayerParameter* v_scale_param = net_param->add_layer();
          v_scale_param->set_type("Power");
          v_scale_param->set_name("v_scale_" + ts);
          v_scale_param->mutable_power_param()->set_scale(this->layer_param_.recurrent_tracker_param().softmax_scale());
          v_scale_param->add_bottom("v4_" + ts);
          v_scale_param->add_top("scaled_v_" + ts);
          
          LayerParameter* v_softmax_param = net_param->add_layer();
          v_softmax_param->set_type("Softmax");
          v_softmax_param->set_name("v_softmax_" + ts);
          v_softmax_param->mutable_softmax_param()->set_axis(3);
          v_softmax_param->add_bottom("scaled_v_" + ts);
          v_softmax_param->add_top("softmax_v_" + ts);
          
          LayerParameter* v_tilde_param = net_param->add_layer();
          v_tilde_param->set_type("MatMult");
          v_tilde_param->set_name("vtilde_" + ts);
          v_tilde_param->add_bottom("softmax_v_" + ts);
          v_tilde_param->add_bottom("overlaps2_" + ts);
          v_tilde_param->add_top("vtilde_" + ts);
        } else {
          LayerParameter* v_tilde_param = net_param->add_layer();
          v_tilde_param->set_type("TrackerMatching");
          v_tilde_param->set_name("vtilde_" + ts);
          v_tilde_param->add_bottom("v4_" + ts);
          v_tilde_param->add_bottom("overlaps2_" + ts);
          v_tilde_param->add_bottom("nseg_" + ts);
          
          v_tilde_param->add_top("vtilde_" + ts);
        }
      }
      
      // Add layers to copy vtilde_{t} 2 times
      //     vtilde1_{t}, vtilde2_{t}
      //{
        LayerParameter* vtilde_split_param = net_param->add_layer();
        vtilde_split_param->set_type("Split");
        vtilde_split_param->set_name("vtilde_split_" + ts);
        vtilde_split_param->add_bottom("vtilde_" + ts);
      
        // for computing output_
        vtilde_split_param->add_top("vtilde1_" + ts);
        // for computing C_{t}
        //vtilde_split_param->add_top("vtilde1_" + ts);
      //}
      
      // Add layers to comput
      // 	m_{t} = \phi_1(v1_{t}) where \phi_1 is
      //	column-wise max
      {
        //1) Reshape from 1 x _N x max_ntrack x max_nseg to _N x 1 x max_ntrack x max_nseg
        LayerParameter* r1_param = net_param->add_layer();
        r1_param->set_type("Reshape");
        r1_param->set_name("reshape1_" + ts);
        r1_param->add_bottom("v1_" + ts);
        r1_param->add_top("vr_" + ts);
        BlobShape* r1_top_blob_shape = r1_param->mutable_reshape_param()->mutable_shape();
        //x input is a T_ x N_ x max_nseg x feature_dim_ array
        r1_top_blob_shape->add_dim(input_blob_shape.dim(1));
        r1_top_blob_shape->add_dim(1);
        r1_top_blob_shape->add_dim(max_ntrack);
        r1_top_blob_shape->add_dim(input_blob_shape.dim(2));
        
        //2) Max pooling
        LayerParameter* max_param = net_param->add_layer();
        max_param->set_type("Pooling");
        max_param->set_name("max_" + ts);
        max_param->mutable_pooling_param()->set_pool(PoolingParameter_PoolMethod_MAX);
        max_param->mutable_pooling_param()->set_kernel_h(max_ntrack);
        max_param->mutable_pooling_param()->set_kernel_w(1);
        max_param->add_bottom("vr_" + ts);
        max_param->add_top("mr_" + ts);
        
        
        //3) Reshape from _N x 1 x 1 x max_nseg to 1 x _N x max_nseg
        LayerParameter* r2_param = net_param->add_layer();
        r2_param->set_type("Reshape");
        r2_param->set_name("reshape2_" + ts);
        r2_param->add_bottom("mr_" + ts);
        r2_param->add_top("m_" + ts);
        BlobShape* r2_top_blob_shape = r2_param->mutable_reshape_param()->mutable_shape();
        r2_top_blob_shape->add_dim(1);
        r2_top_blob_shape->add_dim(input_blob_shape.dim(1));
        r2_top_blob_shape->add_dim(input_blob_shape.dim(2));
      }
      
      
      // Add layers to split m_{t} to m1_{t}, and m2_{t}
      {
        LayerParameter* m_split_param = net_param->add_layer();
        m_split_param->set_type("Split");
        m_split_param->set_name("m_split_" + ts);
        m_split_param->add_bottom("m_" + ts);
        m_split_param->add_top("m1_" + ts);
        m_split_param->add_top("m2_" + ts);
      }
      
    
      // Add layers to comput
      // 	H_{t} =  cont{t} * H_{t-1} + X_{t}^\top (M_{t} X_{t})
      {
        //1) MX_{t} = (M_{t} X_{t})
        LayerParameter* mx_param = net_param->add_layer();
        mx_param->CopyFrom(matmult_param);
        mx_param->set_name("mx_" + ts);
        MatMultParameter* mx_matmult_param = mx_param->mutable_matmult_param();
        mx_matmult_param->add_diagonal_input(true);
        mx_param->add_bottom("m1_" + ts);
        mx_param->add_bottom("x2_" + ts);
        mx_param->add_top("mx_" + ts);
        
        // split mx_{t} to mx1_{t} and mx2_{t}
        LayerParameter* mx_split_param = net_param->add_layer();
        mx_split_param->set_type("Split");
        mx_split_param->set_name("mx_split_" + ts);
        mx_split_param->add_bottom("mx_" + ts);
        mx_split_param->add_top("mx1_" + ts);
        mx_split_param->add_top("mx2_" + ts);
        
        //2) XMX_{t} = X_{t}^\top MX_{t}
        LayerParameter* xmx_param = net_param->add_layer();
        xmx_param->CopyFrom(matmult_param);
        MatMultParameter* xmx_matmult_param = xmx_param->mutable_matmult_param();
        xmx_matmult_param->set_transpose_a(true);
        xmx_param->set_name("xmx_" + ts);
        xmx_param->add_bottom("x3_" + ts);
        xmx_param->add_bottom("mx1_" + ts);
        xmx_param->add_top("xmx_" + ts);
        
        
        //3) H_cont_{t-1} = cont{t} * H_{t-1}
        LayerParameter* h_cont_param = net_param->add_layer();
        h_cont_param->CopyFrom(scalar_param);
        h_cont_param->set_name("h_cont_" + tm1s);
        h_cont_param->add_bottom("h2_" + tm1s);
        h_cont_param->add_bottom("cont2_" + ts);
        h_cont_param->add_top("h_cont_" + tm1s);
        
        
        //4) H_{t} = H_cont_{t-1} + XMX_{t}
        LayerParameter* update_h_param = net_param->add_layer();
        update_h_param->CopyFrom(sum_param);
        update_h_param->set_name("h_" + ts);
        update_h_param->add_bottom("h_cont_" + tm1s);
        update_h_param->add_bottom("xmx_" + ts);
        update_h_param->add_top("h_" + ts);
      }
      
      // Add layers to comput
      // 	C_{t} =  cont{t} * C_{t-1} +  v'_t^\top (M_{t} X_{t})
      {
        //2) XMV_{t} = v'_{t}^\top MX_{t}
        LayerParameter* vmx_param = net_param->add_layer();
        vmx_param->CopyFrom(matmult_param);
        vmx_param->set_name("vmx_" + ts);
        if(this->layer_param_.recurrent_tracker_param().use_vtilde_in_c()) {
          vtilde_split_param->add_top("vtilde2_" + ts);
          vmx_param->add_bottom("vtilde2_" + ts);
        } else {
          v_split_param->add_top("v2_" + ts);
          vmx_param->add_bottom("v2_" + ts);
        }
        vmx_param->add_bottom("mx2_" + ts);
        vmx_param->add_top("vmx_" + ts);
        
        //3) C_cont_{t-1} = cont{t} * C_{t-1}
        LayerParameter* c_cont_param = net_param->add_layer();
        c_cont_param->CopyFrom(scalar_param);
        c_cont_param->set_name("c_cont_" + tm1s);
        c_cont_param->add_bottom("c2_" + tm1s);
        c_cont_param->add_bottom("cont3_" + ts);
        c_cont_param->add_top("c_cont_" + tm1s);
        
        //4) C_{t} = C_cont_{t-1} + VMX_{t}
        LayerParameter* update_c_param = net_param->add_layer();
        update_c_param->CopyFrom(sum_param);
        update_c_param->set_name("c_" + ts);
        update_c_param->add_bottom("c_cont_" + tm1s);
        update_c_param->add_bottom("vmx_" + ts);
        update_c_param->add_top("c_" + ts);
      }
      vtilde_concat_layer.add_bottom("vtilde1_" + ts);
      output_concat_layer.add_bottom("v3_" + ts);
    }  // for (int t = 1; t <= this->T_; ++t)
    
    net_param->add_layer()->CopyFrom(output_concat_layer);
    net_param->add_layer()->CopyFrom(vtilde_concat_layer);
    
    // Debug: Prob some of the outputs
    //for(int i = 0; i < debug_probs.size(); i++) {
    //  LayerParameter* debug_prob_layer = net_param->add_layer();
    //  debug_prob_layer->set_type("Split");
    //  debug_prob_layer->set_name("debug_" + debug_probs[i]);
    //  debug_prob_layer->add_bottom(debug_probs[i]);
    //  debug_prob_layer->add_top("out_" + debug_probs[i]);
    //}
    //
    
  }
  INSTANTIATE_CLASS(RecurrentTrackerLayer);
  REGISTER_LAYER_CLASS(RecurrentTracker);
  
}  // namespace caffe
