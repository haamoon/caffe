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
  //const string debug_probs_arr[] = {"mX_1", "xmX_1", "m_1", "c_1", "c_2", "h_1", "h_2", "W_0", "W_1", "hm1_0", "hm1_1", "m_2"};
  //const vector<string> debug_probs(debug_probs_arr, debug_probs_arr+(sizeof(debug_probs_arr)/sizeof(debug_probs_arr[0])));
  //
  
  // Debug: Prob some of the outputs
  //template <typename Dtype>
  //inline int RecurrentTrackerLayer<Dtype>::ExactNumTopBlobs() const {
  //  return 2 + debug_probs.size(); 
  //}
  
  template <typename Dtype>
  void RecurrentTrackerLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
    names->resize(1);
    (*names)[0] = "W_0";
  }
  
  template <typename Dtype>
  void RecurrentTrackerLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
    names->resize(1);
    (*names)[0] = "W_" + this->int_to_str(this->T_);
  }
  
  template <typename Dtype>
  void RecurrentTrackerLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
    
    const int max_ntrack = this->layer_param_.recurrent_tracker_param().max_ntrack();
    CHECK_GT(max_ntrack, 0) << "max_ntrack must be positive";

    const int feature_dim = this->layer_param_.recurrent_tracker_param().feature_dim();
    CHECK_GT(feature_dim, 0) << "feature_dim must be positive.";
    
    shapes->resize(1);
    //W_t is a 1 x N x max_ntrack x feature_dim matrix
    (*shapes)[0].Clear();
    (*shapes)[0].add_dim(1);
    (*shapes)[0].add_dim(this->N_);
    (*shapes)[0].add_dim(max_ntrack);
    (*shapes)[0].add_dim(feature_dim);
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
    names->resize(2);
    (*names)[0] = "X";
    (*names)[1] = "V";
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
    CHECK_EQ(net_param->input(0).compare("X"), 0);
    const BlobShape input_blob_shape = net_param->input_shape(0);
    
    //Check max_nseg >= max_ntrack
    CHECK_GE(input_blob_shape.dim(2), max_ntrack) << "Number of segments " << input_blob_shape.dim(2) << " should be greater than or equal to the number of track " << max_ntrack;
    
    vector<BlobShape> input_shapes;
    RecurrentInputShapes(&input_shapes);
    CHECK_EQ(2, input_shapes.size());
    net_param->add_input("X_0");
    net_param->add_input_shape()->CopyFrom(input_shapes[0]);
    net_param->add_input("V_0");
    net_param->add_input_shape()->CopyFrom(input_shapes[1]);
    
    LayerParameter* cont_slice_param = net_param->add_layer();
    cont_slice_param->CopyFrom(slice_param);
    cont_slice_param->set_name("cont_slice");
    cont_slice_param->add_bottom("cont");
    cont_slice_param->mutable_slice_param()->set_axis(0);
    
    LayerParameter* X_slice_param = net_param->add_layer();
    X_slice_param->CopyFrom(slice_param);
    X_slice_param->set_name("X_slice");
    X_slice_param->add_bottom("X");

    LayerParameter* V_slice_param = net_param->add_layer();
    V_slice_param->CopyFrom(slice_param);
    V_slice_param->set_name("V_slice");
    V_slice_param->add_bottom("V");

    LayerParameter* Y_slice_param = net_param->add_layer();
    Y_slice_param->CopyFrom(slice_param);
    Y_slice_param->set_name("Y_slice");
    Y_slice_param->add_bottom("Y");

    LayerParameter Y_concat_layer;
    Y_concat_layer.set_name("Y_concat");
    Y_concat_layer.set_type("Concat");
    Y_concat_layer.add_top("Y");
    Y_concat_layer.mutable_concat_param()->set_axis(0);

    for (int t = 1; t <= this->T_; ++t) {
      string tm1s = this->int_to_str(t - 1);
      string ts = this->int_to_str(t);
      
      // Get the slice/row of each matrix of index t
      cont_slice_param->add_top("cont_" + ts);
      X_slice_param->add_top("X_" + ts);
      V_slice_param->add_top("V_" + ts);
      Y_slice_param->add_top("Y_" + ts);



      // Add layer to split
      //     X_t into 4 outputs
      LayerParameter* X_split_param = net_param->add_layer();
      X_split_param->set_type("Split");
      X_split_param->set_name("X_split" + ts);
      X_split_param->add_bottom("X" + ts);
      X_split_param->add_top("X1" + ts);
      X_split_param->add_top("X2" + ts);
      X_split_param->add_top("X3" + ts);
      X_split_param->add_top("X4" + ts);

      // Add layer to compute
      //     XX_t := X_t X_t^\top
      LayerParameter* XX_param = net_param->add_layer();
      XX_param->CopyFrom(matmult_param);
      XX_param->mutable_matmult_param()->set_transpose_b(true);
      XX_param->set_name("XX" + ts);
      XX_param->add_bottom("X1" + ts);
      XX_param->add_bottom("X2" + ts);
      XX_param->add_top("XX" + ts);

      // Add layer to compute
      //     XXpl_t := (X_t X_t^\top + \lambda I)^{-1}
      LayerParameter* XXPL_param = net_param->add_layer();
      XXPL_param->CopyFrom(sum_param);
      XXPL_param->set_type("MatInv");
      XXPL_param->mutable_matinv_param()->set_lambda(this->layer_param_.recurrent_tracker_param().lambda());
      XXPL_param->set_name("XXpl" + ts);
      XXPL_param->add_bottom("XX" + ts);
      XXPL_param->add_top("XXpl" + ts);

      // Add layer to compute
      //     XXplX_t := XXpl_t X_t
      LayerParameter* XXPLX_param = net_param->add_layer();
      XXPLX_param->CopyFrom(matmult_param);
      XXPLX_param->set_name("XXplX" + ts);
      XXPLX_param->add_bottom("XXpl" + ts);
      XXPLX_param->add_bottom("X3" + ts);
      XXPLX_param->add_top("XXplX" + ts);

      // Add layer to compute
      //     Wstart_t := V_t^\top XXplX_t
      LayerParameter* Wstart_param = net_param->add_layer();
      Wstart_param->CopyFrom(matmult_param);
      Wstart_param->mutable_matmult_param()->set_transpose_a(true);
      Wstart_param->set_name("Wstart" + ts);
      Wstart_param->add_bottom("V" + ts);
      Wstart_param->add_bottom("XXplX" + ts);
      Wstart_param->add_top("Wstart" + ts);



      // Add layer to split
      //     W_{t - 1} into 2 outputs
      LayerParameter* W_split_param = net_param->add_layer();
      W_split_param->set_type("Split");
      W_split_param->set_name("W_split_" + tm1s);
      W_split_param->add_bottom("W_" + tm1s);
      W_split_param->add_top("W1_" + tm1s);
      W_split_param->add_top("W2_" + tm1s);

      // Add layer to compute
      //     YmV_{t-1} := Y_{t-1} - V_{t-1}
      LayerParameter* YmV_param = net_param->add_layer();
      YmV_param->CopyFrom(sum_param);
      EltwiseParameter* YmV_elm_param = YmV_param->mutable_eltwise_param();
      YmV_elm_param->add_coeff(1.0);
      YmV_elm_param->add_coeff(-1.0);
      YmV_param->set_name("YmV_" + tm1s);
      YmV_param->add_bottom("Y_" + tm1s);
      YmV_param->add_bottom("V_" + tm1s);
      YmV_param->add_top("YmV_" + tm1s);
      
      // Add layer to compute
      //     XYmV_{t-1} := X_{t-1}^\top YmV_{t-1}
      LayerParameter* XYmV_param = net_param->add_layer();
      XYmV_param->CopyFrom(matmult_param);
      XYmV_param->mutable_matmult_param()->set_transpose_a(true);
      XYmV_param->set_name("XYmV_" + tm1s);
      XYmV_param->add_bottom("X_" + tm1s);
      XYmV_param->add_bottom("YmV_" + tm1s);
      XYmV_param->add_top("XYmV_" + tm1s);

      // Add layer to compute
      //     bXYmV_{t-1} := \beta XYmV_{t-1}
      LayerParameter* bXYmV_param = net_param->add_layer();
      bXYmV_param->CopyFrom(scalar_param);
      bXYmV_param->set_name("bXYmV_" + tm1s);
      bXYmV_param->add_bottom("beta");
      bXYmV_param->add_bottom("XYmV_" + tm1s);
      bXYmV_param->add_top("bXYmV_" + tm1s);

      // Add layer to compute
      //     lW_{t-1} := \lambda W_{t-1}
      LayerParameter* lW_param = net_param->add_layer();
      lW_param->CopyFrom(scalar_param);
      lW_param->set_name("lW_" + tm1s);
      lW_param->add_bottom("lambda");
      lW_param->add_bottom("W1_" + tm1s);
      lW_param->add_top("lW_" + tm1s);

      // Add layer to compute
      //     bXYmVplW_{t-1} := bXYmV_{t-1} + lW_{t-1}
      LayerParameter* bXYmVplW_param = net_param->add_layer();
      bXYmVplW_param->CopyFrom(sum_param);
      bXYmVplW_param->set_name("bXYmVplW_" + tm1s);
      bXYmVplW_param->add_bottom("bXYmV_" + tm1s);
      bXYmVplW_param->add_bottom("lW_" + tm1s);
      bXYmVplW_param->add_top("bXYmVplW_" + tm1s);

      // Add layer to compute
      //     Wcont_t := W_{t-1} + bXYmVplW_{t-1}
      LayerParameter* Wcont_param = net_param->add_layer();
      Wcont_param->CopyFrom(sum_param);
      Wcont_param->set_name("W_" + ts);
      Wcont_param->add_bottom("W_" + tm1s);
      Wcont_param->add_bottom("bXYmVplW_" + tm1s);
      Wcont_param->add_top("W_" + ts);

      // Add layer to compute
      //     Y_t := X_t W_{t-1}^\top
      LayerParameter* Y_param = net_param->add_layer();
      Y_param->CopyFrom(matmult_param);
      Y_param->mutable_matmult_param()->set_transpose_b(true);
      Y_param->set_name("Y_" + ts);
      Y_param->add_bottom("X4_" + ts);
      Y_param->add_bottom("W_" + tm1s);
      Y_param->add_top("Y_" + ts);



      // Finally, switch between Wcont_t or Wstart_t depending on whether cont is true
      LayerParameter* W_param = net_param->add_layer();
      W_param->set_type("Switch");
      W_param->set_name("W_" + ts);
      W_param->mutable_switch_param()->set_axis(2);
      W_param->add_bottom("Wstart_" + ts);
      W_param->add_bottom("Wcont_" + ts);
      W_param->add_bottom("cont_" + ts);
      W_param->add_top("W_" + ts);

      Y_concat_layer.add_bottom("Y_" + ts);
    }

    net_param->add_layer()->CopyFrom(Y_concat_layer);
    
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
