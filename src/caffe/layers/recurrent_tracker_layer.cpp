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
  const string debug_probs_arr[] = {"Wcont_0", "Wcont_1"};
  const vector<string> debug_probs(debug_probs_arr, debug_probs_arr+(sizeof(debug_probs_arr)/sizeof(debug_probs_arr[0])));
  //
  
  // Debug: Prob some of the outputs
  //template <typename Dtype>
  //inline int RecurrentTrackerLayer<Dtype>::ExactNumTopBlobs() const {
  //  return 1 + debug_probs.size();
  //}
  
  template <typename Dtype>
  void RecurrentTrackerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
    max_ntrack_ = bottom[1]->shape(2);
    max_nseg_ = bottom[0]->shape(2);
    feature_dim_ = bottom[0]->shape(3);
    RecurrentLayer<Dtype>::LayerSetUp(bottom, top);
  }
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
    
    //const int max_ntrack = this->layer_param_.recurrent_tracker_param().max_ntrack();
    CHECK_GT(max_ntrack_, 0) << "max_ntrack must be positive";
    
    //const int feature_dim = this->layer_param_.recurrent_tracker_param().feature_dim();
    CHECK_GT(feature_dim_, 0) << "feature_dim must be positive.";
    
    shapes->resize(1);
    //W_t is a 1 x N x max_ntrack x feature_dim matrix
    (*shapes)[0].Clear();
    (*shapes)[0].add_dim(1);
    (*shapes)[0].add_dim(this->N_);
    (*shapes)[0].add_dim(max_ntrack_);
    (*shapes)[0].add_dim(feature_dim_);
  }
  
  template <typename Dtype>
  void RecurrentTrackerLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
    names->resize(1);
    (*names)[0] = "Y";
    // Debug: Prob some of the outputs
    names->resize(1 + debug_probs.size());
    for(int i = 0; i < debug_probs.size(); i++) {
      (*names)[i + 1] = ("out_" + debug_probs[i]);
    }
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
    //const int max_ntrack = this->layer_param_.recurrent_tracker_param().max_ntrack();
    const double lambda = this->layer_param_.recurrent_tracker_param().lambda();
    const double alpha = this->layer_param_.recurrent_tracker_param().alpha();
    
    // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
    // use to save redundant code.
    LayerParameter matmult_param;
    matmult_param.set_type("MatMult");
    
    LayerParameter sum_param;
    sum_param.set_type("Eltwise");
    sum_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
    
    LayerParameter slice_param;
    slice_param.set_type("Slice");
    slice_param.mutable_slice_param()->set_axis(0);
    
    LayerParameter power_param;
    power_param.set_type("Power");


    //TODO: currently feature_dim_ can not be set because this
    //fucntion is defined as constant. So, I set it mannually
    //in RecurrentInputShapes proto.
    //initialize feature_dim_ before calling RecurrentInputShapes
    //x input is a T_ x N_ x max_nseg x feature_dim_ array
    //CHECK_GE(net_param->input_size(), 1);
    //CHECK_EQ(net_param->input(0).compare("X"), 0); // 
    //const BlobShape input_blob_shape = net_param->input_shape(0);

    //Check max_nseg >= max_ntrack
    //CHECK_GE(input_blob_shape.dim(2), max_ntrack) << "Number of segments " << input_blob_shape.dim(2) << " should be greater than or equal to the number of track " << max_ntrack;

    
    
    //vector<BlobShape> input_shapes;
    //RecurrentInputShapes(&input_shapes);
    //CHECK_EQ(1, input_shapes.size());
    //net_param->add_input("W_0");
    //net_param->add_input_shape()->CopyFrom(input_shapes[0]);

    LayerParameter* cont_slice_param = net_param->add_layer();
    cont_slice_param->CopyFrom(slice_param);
    cont_slice_param->set_name("cont_slice");
    cont_slice_param->add_bottom("cont");
    
    LayerParameter* X_slice_param = net_param->add_layer();
    X_slice_param->CopyFrom(slice_param);
    X_slice_param->set_name("X_slice");
    X_slice_param->add_bottom("X");

    LayerParameter* V_slice_param = net_param->add_layer();
    V_slice_param->CopyFrom(slice_param);
    V_slice_param->set_name("V_slice");
    V_slice_param->add_bottom("V");

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



      // Add layer to split
      //     X_t into 4 outputs
      //LayerParameter* X_split_param = net_param->add_layer();
      //X_split_param->set_type("Split");
      //X_split_param->set_name("X_split" + ts);
      //X_split_param->add_bottom("X_" + ts);
      //X_split_param->add_top("X1_" + ts);
      //X_split_param->add_top("X2_" + ts);
      //X_split_param->add_top("X3_" + ts);
      //X_split_param->add_top("X4_" + ts);

      // Add layer to compute
      //     XX_t := X_t X_t^\top
      LayerParameter* XX_param = net_param->add_layer();
      XX_param->CopyFrom(matmult_param);
      XX_param->set_name("XX_" + ts);
      XX_param->add_bottom("X_" + ts);
      XX_param->mutable_matmult_param()->set_transpose_b(true);
      XX_param->add_bottom("X_" + ts);
      XX_param->add_top("XX_" + ts);

      // Add layer to compute
      //     XXpl_t := (X_t X_t^\top + \lambda I)^{-1}
      LayerParameter* XXpl_param = net_param->add_layer();
      XXpl_param->set_type("MatInv");
      XXpl_param->mutable_matinv_param()->set_lambda(this->layer_param_.recurrent_tracker_param().lambda());
      XXpl_param->set_name("XXpl_" + ts);
      XXpl_param->add_bottom("XX_" + ts);
      XXpl_param->add_top("XXpl_" + ts);

      // Add layer to compute
      //     VXXpl_t := V_t XXpl_t
      LayerParameter* VXXpl_param = net_param->add_layer();
      VXXpl_param->CopyFrom(matmult_param);
      VXXpl_param->set_name("VXXpl_" + ts);
      VXXpl_param->add_bottom("V_" + ts);
      VXXpl_param->add_bottom("XXpl_" + ts);
      VXXpl_param->add_top("VXXpl_" + ts);

      // Add layer to compute
      //     Wstart_{t - 1} := VXXpl_t X_t
      LayerParameter* Wstart_param = net_param->add_layer();
      Wstart_param->CopyFrom(matmult_param);
      Wstart_param->set_name("Wstart_" + tm1s);
      Wstart_param->add_bottom("VXXpl_" + ts);
      Wstart_param->add_bottom("X_" + ts);
      Wstart_param->add_top("Wstart_" + tm1s);
      
      /*
      // Add layer to split
      //     W_{t - 1} into 2 outputs
      LayerParameter* W_split_param = net_param->add_layer();
      W_split_param->set_type("Split");
      W_split_param->set_name("W_split_" + tm1s);
      W_split_param->add_bottom("W_" + tm1s);
      W_split_param->add_top("W1_" + tm1s);
      W_split_param->add_top("W2_" + tm1s);
      */

      // Switch between Wstart_{t - 1} or W_{t - 1} depending on whether cont is true
      LayerParameter* Wcont_param = net_param->add_layer();
      Wcont_param->set_type("Switch");
      Wcont_param->set_name("Wcont_" + tm1s);
      Wcont_param->mutable_switch_param()->set_axis(2); // only consider dimensions starting with 2
      Wcont_param->add_bottom("Wstart_" + tm1s);
      Wcont_param->add_bottom("W_" + tm1s);
      Wcont_param->add_bottom("cont_" + ts);
      Wcont_param->add_top("Wcont_" + tm1s);

      // Add layer to compute
      //     Y_t := Wcont_{t-1} X_t^\top
      LayerParameter* Y_param = net_param->add_layer();
      Y_param->CopyFrom(matmult_param);
      Y_param->set_name("Y_" + ts);
      Y_param->add_bottom("Wcont_" + tm1s);
      Y_param->add_bottom("X_" + ts);
      Y_param->add_top("Y_" + ts);
      Y_param->mutable_matmult_param()->set_transpose_b(true);

      // Add layer to compute
      //     cW_{t-1} := (1 - \alpha\lambda) Wcont_{t-1}
      //LayerParameter* cW_param = net_param->add_layer();
      //cW_param->CopyFrom(power_param);
      //cW_param->set_name("cW_" + tm1s);
      //cW_param->mutable_power_param()->set_scale(1 - alpha * lambda);
      //cW_param->add_bottom("Wcont_" + tm1s);
      //cW_param->add_top("cW_" + tm1s);

      // Add layer to compute
      //     YmV_t := Y_t - V_t
      LayerParameter* YmV_param = net_param->add_layer();
      YmV_param->CopyFrom(sum_param);
      YmV_param->set_name("YmV_" + ts);
      YmV_param->add_bottom("Y_" + ts);
      YmV_param->add_bottom("V_" + ts);
      YmV_param->add_top("YmV_" + ts);
      EltwiseParameter* YmV_elm_param = YmV_param->mutable_eltwise_param();
      YmV_elm_param->add_coeff(1.0);
      YmV_elm_param->add_coeff(-1.0);
      
      // Add layer to compute
      //     YmVX_t := YmV_t X_t
      LayerParameter* YmVX_param = net_param->add_layer();
      YmVX_param->CopyFrom(matmult_param);
      YmVX_param->set_name("YmVX_" + ts);
      YmVX_param->add_bottom("YmV_" + ts);
      YmVX_param->add_bottom("X_" + ts);
      YmVX_param->add_top("YmVX_" + ts);

      // Add layer to compute
      //     aYmVX_t := \alpha YmVX_t
      //LayerParameter* aYmVX_param = net_param->add_layer();
      //aYmVX_param->CopyFrom(power_param);
      //aYmVX_param->set_name("aYmVX_" + ts);
      //aYmVX_param->mutable_power_param()->set_scale(alpha);
      //aYmVX_param->add_bottom("YmVX_" + ts);
      //aYmVX_param->add_top("aYmVX_" + ts);

      // Add layer to compute
      //     W_t := (1 - alpha * lambda) Wcont_{t-1} - alpha YmVX_t
      LayerParameter* W_param = net_param->add_layer();
      W_param->CopyFrom(sum_param);
      W_param->set_name("W_" + ts);
      W_param->add_bottom("Wcont_" + tm1s);
      W_param->add_bottom("YmVX_" + ts);
      W_param->add_top("W_" + ts);
      EltwiseParameter* W_elm_param = W_param->mutable_eltwise_param();
      W_elm_param->add_coeff(1 - alpha * lambda);
      W_elm_param->add_coeff(-alpha);
      
      Y_concat_layer.add_bottom("Y_" + ts);
    }

    net_param->add_layer()->CopyFrom(Y_concat_layer);
    
    // Debug: Prob some of the outputs
    for(int i = 0; i < debug_probs.size(); i++) {
      LayerParameter* debug_prob_layer = net_param->add_layer();
      debug_prob_layer->set_type("Split");
      debug_prob_layer->set_name("debug_" + debug_probs[i]);
      debug_prob_layer->add_bottom(debug_probs[i]);
      debug_prob_layer->add_top("out_" + debug_probs[i]);
    }
    //
    
  }
  INSTANTIATE_CLASS(RecurrentTrackerLayer);
  REGISTER_LAYER_CLASS(RecurrentTracker);
  
}  // namespace caffe
