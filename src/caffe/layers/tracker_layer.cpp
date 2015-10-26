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
    
    template <typename Dtype>
    void TrackerLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
        names->resize(2);
        (*names)[0] = "c_0";
        (*names)[1] = "h_0";
    }
    
    template <typename Dtype>
    void TrackerLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
        names->resize(2);
        (*names)[0] = "c_" + this->int_to_str(this->T_);
        (*names)[1] = "h_" + this->int_to_str(this->T_);
    }
    
    template <typename Dtype>
    void TrackerLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {

        const int feature_dim = this->layer_param_.tracker_param().feature_dim(); 
        CHECK_GT(feature_dim, 0) << "feature_dim must be positive.";
        
        const int num_track = this->layer_param_.tracker_param().num_track();
        CHECK_GT(num_track, 0) << "num_track must be positive";
        
        shapes->resize(2);
        //C_t is a d x num_track matrix
        (*shapes)[0].Clear();
        (*shapes)[0].add_dim(1);  // a single timestep
        (*shapes)[0].add_dim(this->N_);
        (*shapes)[0].add_dim(feature_dim);
        (*shapes)[0].add_dim(num_track);
        
        //H_t is a dxd matrix
        (*shapes)[1].Clear();
        (*shapes)[1].add_dim(1);  // a single timestep
        (*shapes)[1].add_dim(this->N_);
        (*shapes)[1].add_dim(feature_dim);
        (*shapes)[1].add_dim(feature_dim);
    }
    
    template <typename Dtype>
    void TrackerLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
        names->resize(1);
        (*names)[0] = "v";
    }
    
    template <typename Dtype>
    void TrackerLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {

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
        
        //TODO: currently feature_dim_ can not be set because this
        //fucntion is defined as constant. So, I set it mannually
        //in proto.
        //initialize feature_dim_ before calling RecurrentInputShapes
        //x input is a T_ x N_ x num_seg x feature_dim_ array
        //CHECK_GE(net_param->input_size(), 1);
        //CHECK_EQ(net_param->input(0).compare("x"), 0);
        //const BlobShape input_blob_shape = net_param->input_shape(0);
        //feature_dim_ = input_blob_shape.dim(input_blob_shape.dim_size() - 1);
        
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
        
        LayerParameter output_concat_layer;
        output_concat_layer.set_name("v_concat");
        output_concat_layer.set_type("Concat");
        output_concat_layer.add_top("v");
        output_concat_layer.mutable_concat_param()->set_axis(0);
        
        for (int t = 1; t <= this->T_; ++t) {
            string tm1s = this->int_to_str(t - 1);
            string ts = this->int_to_str(t);
            
            cont_slice_param->add_top("cont_" + ts);
            x_slice_param->add_top("x_" + ts);
            
            
            // Add layers to compute
            //     hm1_{t-1} := h_{t-1}^{-1}
            {
                LayerParameter* hm1_param = net_param->add_layer();
                hm1_param->set_type("MatInv");
                hm1_param->set_name("hm1_" + tm1s);
                hm1_param->add_bottom("h_" + tm1s);
                hm1_param->add_top("hm1_" + tm1s);
            }
            
            // Add layers to compute
            //     w_{t-1} := hm1_{t-1} c_{t-1}
            {
                LayerParameter* w_param = net_param->add_layer();
                w_param->CopyFrom(matmult_param);
                w_param->set_name("w_" + tsm);
                w_param->add_bottom("hm1_" + tm1s);
                w_param->add_bottom("c_" + tm1s)
                w_param->add_top("w_" + tm1s);
            }
            
            // Add layer to compute
            //     v_{t} = X_{t} w_{t-1}
            {
                LayerParameter* v_param = net_param->add_layer();
                v_param->CopyFrom(matmult_param);
                v_param->set_name("v_" + ts);
                v_param->add_bottom("x_" + ts);
                v_param->add_bottom("w_" + tm1s);
                v_param->add_top("v_" + ts);
            }
            
            // Add layers to compute
            //     v_cont_{t} = v_{t} * cont{t}
            {
                LayerParameter* v_cont_param = net_param->add_layer();
                v_cont_param->set_type("Switch");
                v_cont_param->set_name("v_cont_" + ts);
                v_cont_param->add_bottom("v_" + ts);
                v_cont_param->add_bottom("cont_" + ts);
                v_cont_param->add_top("v_cont_" + ts);
            }
            
            // Add layers to compute
            //     m_{t} =
            {
                LayerParameter* v_cont_param = net_param->add_layer();
                v_cont_param->set_type("Switch");
                v_cont_param->set_name("v_cont_" + ts);
                v_cont_param->add_bottom("v_" + ts);
                v_cont_param->add_bottom("cont_" + ts);
                v_cont_param->add_top("v_cont_" + ts);
            }
            
            
            output_concat_layer.add_bottom("v_cont_" + ts);
        }  // for (int t = 1; t <= this->T_; ++t)
        
        net_param->add_layer()->CopyFrom(output_concat_layer);
    }
    
    INSTANTIATE_CLASS(TrackerLayer);
    REGISTER_LAYER_CLASS(Tracker);
    
}  // namespace caffe
