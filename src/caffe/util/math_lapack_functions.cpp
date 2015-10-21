#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/math_lapack_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
	template <>
	void caffe_cpu_inverse<float>(const int n, float* X) {
    	/*  
    		See http://itf.fys.kuleuven.be/~rob/computer/lapack_wrapper/
    	    Calculates the inverse of the n*n matrix X: Y = Y^-1
    	    Does not change the value of X, unless Y=X
        */
        
        int info=0;
        
        
        /*  We need to store the pivot matrix obtained by the LU factorisation  */
        int *ipiv;
        ipiv= (int*) malloc(n*sizeof(int));
        if (ipiv==NULL) {
        		LOG(FATAL) << "malloc failed in matrix_invert";
        }
        
        /*  Turn X into its LU form, store pivot matrix  */
        info = clapack_sgetrf (CblasRowMajor, n, n, X, n, ipiv);
        
        /*  Don't bother continuing when illegal argument (info<0) or singularity (info>0) occurs  */
        if (info!=0) LOG(FATAL) << "Matrix inversion failed: " << info;
                
        /*  Feed this to the lapack inversion routine.  */
        info = clapack_sgetri (CblasRowMajor, n, X, n, ipiv);
        
        /*  Cleanup */
        free(ipiv);
}


template <>
	void caffe_cpu_inverse<double>(const int n, double* X) {
    	/*  
    		See http://itf.fys.kuleuven.be/~rob/computer/lapack_wrapper/
    	    Calculates the inverse of the n*n matrix X: Y = X^-1
    	    Does not change the value of X, unless Y=X
        */
        
        int info=0;
        
        /*  We need to store the pivot matrix obtained by the LU factorisation  */
        int *ipiv;
        ipiv= (int*) malloc(n*sizeof(int));
        if (ipiv==NULL) {
        		LOG(FATAL) << "malloc failed in matrix_invert";
        }
        
        /*  Turn X into its LU form, store pivot matrix  */
        info = clapack_dgetrf (CblasRowMajor, n, n, X, n, ipiv);
        
        /*  Don't bother continuing when illegal argument (info<0) or singularity (info>0) occurs  */
        if (info!=0) LOG(FATAL) << "Matrix inversion failed: " << info;
                
        /*  Feed this to the lapack inversion routine.  */
        info = clapack_dgetri (CblasRowMajor, n, X, n, ipiv);
        
        /*  Cleanup  */
        free(ipiv);
}

}  // namespace caffe
