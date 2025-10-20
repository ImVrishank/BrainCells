#include <torch/torch.h>
#include <iostream>
#include<utility>


torch::Tensor relu(torch::Tensor z, bool derivative = false){
    if (derivative) {
        return (z>0).toType(torch::kFloat32); 
    }
    return torch::maximum(z, torch::scalar_tensor(0.0).to(z.dtype()));
}

class Layer{
    public: 
        torch::Tensor weights;
        torch::Tensor bias;
        torch::Tensor activation;
        torch::Tensor z;
        torch::Tensor error;

        Layer(int input_size, int output_size){
            this->weights = torch::rand({output_size, input_size}) * 0.01;
            this->bias = torch::rand({1, output_size});
            this->activation = torch::zeros({1, output_size});
            this->z = torch::zeros({1, output_size});
            this->error = torch::zeros({1, output_size});
        }

        torch::Tensor forward(torch::Tensor inputs){
            this->z = torch::matmul(inputs, this->weights.t()) + this->bias;
            this->activation = relu(this->z);
            return this->activation;
        }

        std::pair<torch::Tensor, torch::Tensor> calc_backpropagation(torch::Tensor next_layer_errors = torch.Tensor(), torch::Tensor next_layer_weights = torch.Tensor(), bool last_layer = false, torch::Tensor Y = torch.Tensor()){
            if (last_layer){
                this->error = (this->activation - Y) * relu(this->z, derivative=true);
            }

            else{
                this->error = (torch::matmul(next_layer_errors, next_layer_weights)) * relu(this->z, true);
            }

            return std::make_pair(this->error, this->weights);

        }

        void SGD(torch::Tensor inputs, double lr){
            auto grad_w = torch::matmul(this->error.t(), inputs);
            auto grad_b = this->error.sum(0, true);

            this->weights = this->weights - lr * grad_w;
            this->bias = this->bias - lr * grad_b;
        }
};

int main() {
    
}