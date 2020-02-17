/////////////////////////////////////////////////////////////////////////////////////////
//  Custom_Layers.cpp
//  Feb. 2020 - J. Hill
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//  implements the forward and bachward passess of
//  BasisRotation, Projection, and Vectorizer
//  layers defined in Custom_Layers.py
/////////////////////////////////////////////////////////////////////////////////////////

#include <torch/torch.h>
#include <torch/extension.h>
#include <stdlib.h>
#include <vector>


/////////////////////////////////////////////////////////////////////////////////////////
//  BasisRotation forward and backward functions
/////////////////////////////////////////////////////////////////////////////////////////
at::Tensor BasisRotation_forward(torch::Tensor input, torch::Tensor R)
{
  auto input_dims = input.sizes();
  auto operator_dims = R.sizes();

  int64_t output_channels = operator_dims[0];
  int64_t batch_size = input_dims[0];
  int64_t input_channels = input_dims[1];
  int64_t inner_dim = input_dims[2];
  
  auto options = torch::TensorOptions().dtype(torch::kFloat64);  
  auto output = torch::zeros({batch_size, output_channels, inner_dim, inner_dim}, options);

  //accessors are both 4-dimensional because R indices are (output channel, input channel, row, column)
  //input indices are (batch index, input channel, row, column)
  //output indices are (batch index, output channel, row, column)

  //accessors may be used if dynamic dispatch is too expensive for high-rank tensor operations
  //auto R_a = R.accessor<double,4>();
  //auto input_a = input.accessor<double,4>();
  //auto output_a = output.accessor<double,4>();
  
  for (int n = 0; n < batch_size; n++)
    {
      for (int i = 0; i < output_channels; i++)
	{
	  for (int j = 0; j < input_channels; j++)
	    {
	      //auto O = R_a[i][j];
	      //auto mat = input_a[n][j];
	      
	      output[n][i] += at::chain_matmul({R[i][j], input[n][i], at::transpose(R[i][j],0,1)});
	    }
	}
    }

  return output;
}//end of BasisRotation_forward

std::vector< at::Tensor > BasisRotation_backward(torch::Tensor input, torch::Tensor grad_output, torch::Tensor R)
{
  auto input_dims = input.sizes();
  auto operator_dims = R.sizes();

  int64_t output_channels = operator_dims[0];
  int64_t batch_size = input_dims[0];
  int64_t input_channels = input_dims[1];
  int64_t inner_dim = input_dims[2];

  int64_t i,ii,j,jj,k,kk,n;
  double coeff;
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  
  auto grad_input = torch::zeros_like(input, options);
  auto grad_R = torch::zeros({output_channels, input_channels, inner_dim, inner_dim}, options);

  auto R_a = R.accessor<double,4>();
  auto grad_R_a = grad_R.accessor<double,4>();
  
  auto input_a = input.accessor<double,4>();
  auto grad_input_a = grad_input.accessor<double,4>();
  auto grad_output_a = grad_output.accessor<double,4>();

  //compute partial derivatives of outputs w.r.t. the inputs and operator
  for (n = 0; n < batch_size; n++)
    {
      for (i = 0; i < output_channels; i++)
	{	  
	  for (ii = 0; ii < input_channels; ii++)
	    {
	      for (j = 0; j < inner_dim; j++)
		{
		  for (jj = 0; jj < inner_dim; jj++)
		    {
		      for (k = 0; k < inner_dim; k++)
			{
			  for (kk = 0; kk < inner_dim; kk++)
			    {
			      grad_input_a[n][ii][j][k] += R_a[i][ii][kk][j]*R_a[i][ii][jj][k]*grad_output_a[n][i][kk][jj];
			      grad_R_a[i][ii][j][k] += input_a[n][ii][k][jj]*R_a[i][ii][kk][jj]*grad_output_a[n][i][j][kk];			      
			    }
			}
		    }
		}
	    }
	}
    }
  
  return {grad_input, grad_R};
}//end of BasisRotation_backward

/////////////////////////////////////////////////////////////////////////////////////////
// Projection forward and backward functions
/////////////////////////////////////////////////////////////////////////////////////////

at::Tensor Projection_forward(torch::Tensor input, torch::Tensor P, torch::Tensor Pt)
{
  auto input_dims = input.sizes();
  auto operator_dims = P.sizes();
  auto operator_t_dims = Pt.sizes();

  // operator and input matrix dimensions are based on the assumption of rank-4 tensors

  int64_t output_channels = operator_dims[0];
  int64_t batch_size = input_dims[0];
  int64_t input_channels = input_dims[1];
  int64_t M = input_dims[2];
  int64_t N = input_dims[3];
  int64_t Mp = operator_dims[2];
  int64_t Np = operator_t_dims[3];

  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  auto output = torch::zeros({batch_size, output_channels, Mp, Np}, options);
  
  for (int n = 0; n < batch_size; n++)
    {
      for (int i = 0; i < output_channels; i++)
	{
	  for (int j = 0; j < input_channels; j++)
	    {
	      //auto O = R_a[i][j];
	      //auto mat = input_a[n][j];
	      
	      output[n][i] += at::chain_matmul({P[i][j], input[n][j], Pt[i][j]});
	    }
	}
    }

  return output;
}//end of Projection_forward

std::vector< at::Tensor > Projection_backward(torch::Tensor input, torch::Tensor grad_output, torch::Tensor P, torch::Tensor Pt)
{
  auto input_dims = input.sizes();
  auto operator_dims = P.sizes();
  auto operator_t_dims = Pt.sizes();

  int64_t output_channels = operator_dims[0];
  int64_t batch_size = input_dims[0];
  int64_t input_channels = input_dims[1];
  int64_t M = input_dims[2];
  int64_t N = input_dims[3];
  int64_t Mp = operator_dims[2];
  int64_t Np = operator_t_dims[3];

  int64_t i,ii,j,jp,k,kp,n;
  double coeff;
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  
  auto grad_input = torch::zeros_like(input, options);
  auto grad_P = torch::zeros({output_channels, input_channels, Mp, M}, options);
  auto grad_Pt = torch::zeros({output_channels, input_channels, N, Np}, options);

  auto P_a = P.accessor<double,4>();
  auto Pt_a = Pt.accessor<double,4>();
  auto grad_P_a = grad_P.accessor<double,4>();
  auto grad_Pt_a = grad_Pt.accessor<double,4>();

  auto input_a = input.accessor<double,4>();
  auto grad_input_a = grad_input.accessor<double,4>();
  auto grad_output_a = grad_output.accessor<double,4>();

  //compute partial derivatives of outputs w.r.t. the inputs and operators
  for (n = 0; n < batch_size; n++)
    {
      for (i = 0; i < output_channels; i++)
	{
	  for (ii = 0; ii < input_channels; ii++)
	    {
	      for (jp = 0; jp < Mp; jp++)
		{
		  for (j = 0; j < M; j++)
		    {
		      for (k = 0; k < N; k++)
			{
			  for (kp = 0; kp < Np; kp++)
			    {
			      grad_input_a[n][ii][j][k] += P_a[i][ii][jp][j]*Pt_a[i][ii][k][kp]*grad_output_a[n][i][jp][kp];
			      grad_P_a[i][ii][jp][j] += input_a[n][ii][j][k]*Pt_a[i][ii][k][kp]*grad_output_a[n][i][jp][kp];
			      grad_Pt_a[i][ii][k][kp] += P_a[i][ii][jp][j]*input_a[n][ii][j][k]*grad_output_a[n][i][jp][kp];
			    }
			}
		    }
		}
	    }
	}
    }
  
  return {grad_input, grad_P, grad_Pt};
}//end of Projection_backward



/////////////////////////////////////////////////////////////////////////////////////////
// Vectorizer forward and backward functions
/////////////////////////////////////////////////////////////////////////////////////////


at::Tensor Vectorizer_forward(torch::Tensor input, torch::Tensor v)
{
  auto input_dims = input.sizes();
  auto vector_dims = v.sizes();

  int64_t output_channels = vector_dims[0];
  int64_t batch_size = input_dims[0];
  int64_t input_channels = input_dims[1];
  int64_t M = input_dims[2];
  int64_t N = input_dims[3];

  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  auto output = torch::zeros({batch_size, output_channels, M}, options);
  
  for (int n = 0; n < batch_size; n++)
    {
      for (int i = 0; i < output_channels; i++)
	{
	  for (int j = 0; j < input_channels; j++)
	    {
	      output[n][i] += at::linear(input[n][j], v[i][j]);
	    }
	}
    }

  return output;
}//endof Vectorizer_forward

std::vector< at::Tensor > Vectorizer_backward(torch::Tensor input, torch::Tensor grad_output, torch::Tensor v)
{
  auto input_dims = input.sizes();
  auto vector_dims = v.sizes();
  
  int64_t output_channels = vector_dims[0];
  int64_t batch_size = input_dims[0];
  int64_t input_channels = input_dims[1];
  int64_t M = input_dims[2];
  int64_t N = input_dims[3];

  int64_t i,ii,j,k,n;
  double coeff;
  auto options = torch::TensorOptions().dtype(torch::kFloat64);

  auto grad_input = torch::zeros_like(input, options);
  auto grad_v = torch::zeros({output_channels, input_channels, N}, options);

  auto v_a = v.accessor<double,3>();
  auto grad_v_a = grad_v.accessor<double,3>();

  auto input_a = input.accessor<double,4>();
  auto grad_input_a = grad_input.accessor<double,4>();
  auto grad_output_a = grad_output.accessor<double,3>();

  //compute partial derivatives of outputs w.r.t. the inputs and vector
  for (n = 0; n < batch_size; n++)
    {
      for (i = 0; i < output_channels; i++)
	{	  
	  for (ii = 0; ii < input_channels; ii++)
	    {
	      for (j = 0; j < M; j++)
		{
		  for (k = 0; k < N; k++)
		    {
		      grad_input_a[n][ii][j][k] += v_a[i][ii][k]*grad_output_a[n][i][j];
		      grad_v_a[i][ii][k] += input_a[n][ii][j][k]*grad_output_a[n][i][j];
		    }
		}
	    }
	}
    }
  
  return {grad_input, grad_v};
}//end of Vectorizer_backward




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("BasisRotation_forward", &BasisRotation_forward, "BasisRotation forward");
  m.def("BasisRotation_backward", &BasisRotation_backward, "BasisRotation backward");
  
  m.def("Projection_forward", &Projection_forward, "Projection forward");
  m.def("Projection_backward", &Projection_backward, "Projection backward");

  m.def("Vectorizer_forward", &Vectorizer_forward, "Vectorizer forward");
  m.def("Vectorizer_backward", &Vectorizer_backward, "Vectorizer backward");
}
