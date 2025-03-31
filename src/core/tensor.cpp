#include <unordered_set>
#include "core/tensor.hh"
#include "core/cudaException.hh"
#include "core/ops.hh"
#include "cuda/ops.cuh"

Tensor::Tensor(
    DeviceType device, const std::vector<int>& shape, std::shared_ptr<float> data, std::shared_ptr<float> grad
): device_(device), shape_(shape){
    storage_ = {data, grad, numel()}; 
    allocate_storage();
    calculate_strirde();
}

Tensor::Ptr Tensor::create(const std::vector<int>& shape, DeviceType device) {
    auto tensor = std::make_shared<Tensor>(device, shape, nullptr, nullptr);
    return tensor;
}

Tensor::Ptr Tensor::create(
    const std::vector<float>& data, const std::vector<int> &shape, DeviceType device
){
    auto tensor = create(shape, device);

    if (device == DeviceType::CPU){
        std::copy(data.begin(), data.end(), tensor->storage_.data.get());
    } else {
        cudaMemcpy(
            tensor->storage_.data.get(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice
        );
        CudaException::throwIfCUDAErrorsOccurred("Failed tensor creation on device!");
    }
    return tensor;
}

Tensor::~Tensor(){
    storage_.data.reset();
    storage_.grad.reset();
}

void Tensor::allocate_storage(){
    const size_t elements = numel();
    storage_.size = elements;

    if (device_ == DeviceType::CPU){
        float *arr = new float[elements];
        storage_.data = std::shared_ptr<float>(
            arr, [](float *ptr){delete[] ptr;}
        );
        for(size_t i = 0; i < elements; ++i){arr[i] = 0.0f;}
    }else if (device_ == DeviceType::CUDA){
        float *d_mem = nullptr;
        const size_t bytes = elements * sizeof(float);

        cudaMalloc((void **)&d_mem, bytes);
        cudaMemset(d_mem, 0, bytes);
        CudaException::throwIfCUDAErrorsOccurred("Cannot allocate CUDA memory!");
        storage_.data = std::shared_ptr<float>(
            d_mem, [](float *ptr){cudaFree(ptr);}
        );
    }
}

size_t Tensor::numel() const noexcept{
    if (shape_.empty()) return 0;
    size_t count = 1;
    for (int dim : shape_) count *= dim;
    return count;
}

void Tensor::calculate_strirde(){
    stride_.resize(shape_.size());
    if (shape_.empty()) return;
    stride_.back() = 1;
    for (int i = shape_.size() - 2; i >= 0; --i){
        stride_[i] = stride_[i+1] * shape_[i+1];
    }
}

Tensor::Ptr Tensor::to(DeviceType target_device){
    auto result = create(shape_, target_device);
    if (device_ == DeviceType::CPU && target_device == DeviceType::CUDA){
        cudaMemcpy(
            result->data(), data(), numel() * sizeof(float), cudaMemcpyHostToDevice
        );
        if (grad()){
            result->init_grad();
            cudaMemcpy(
                result->grad(), grad(), numel() * sizeof(float), cudaMemcpyHostToDevice        
            );

        }
        CudaException::throwIfCUDAErrorsOccurred("Failed load to device!");
    }else if (device_ == DeviceType::CUDA && target_device == DeviceType::CPU){
        cudaMemcpy(
            result->data(), data(), numel() * sizeof(float), cudaMemcpyDeviceToHost
        );
        if (grad()){
            result->init_grad();
            cudaMemcpy(
                result->grad(), grad(), numel() * sizeof(float), cudaMemcpyDeviceToHost        
            );

        }
        CudaException::throwIfCUDAErrorsOccurred("Failed load to host!");
    }
    return result;
}

size_t Tensor::linear_index(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Invalid number of indices");
    }

    size_t idx = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        idx += indices[i] * stride_[i];
    }
    return idx;
}

void Tensor::init_grad() const{
    if (!grad_initialized_){
        const size_t elements = numel();
        if (elements == 0) return;
        if (device_ == DeviceType::CPU){
            storage_.grad = std::shared_ptr<float>(
                new float[elements], [](float* ptr){delete[] ptr;}
            );
            std::fill(grad(), grad() + elements, 0.0f);
        } else {
            float* d_grad;
            cudaMalloc(&d_grad, elements * sizeof(float));
            cudaMemset(d_grad, 0, elements * sizeof(float));
            CudaException::throwIfCUDAErrorsOccurred("Init grad failed!");
            storage_.grad = std::shared_ptr<float>(d_grad, [](float *ptr){cudaFree(ptr);});
        }
        grad_initialized_ = true;
    }
}

void Tensor::zero_grad(){
    if (grad_initialized_){
        const size_t elements = numel();
        if (device_ == DeviceType::CPU)
            std::fill(grad(), grad() + elements, 0.0f);
        else
            cudaMemset(grad(), 0, elements * sizeof(float));
    }
}

void Tensor::backward(const Tensor::Ptr& upstream_grad){
    std::vector<Ptr> topo;
    std::unordered_set<Ptr> visited;

    std::function<void(const Ptr&)> build_topo = [&](const Ptr& tensor){
        if (!visited.count(tensor)){
            visited.insert(tensor);

            for (const auto& child : tensor->children_)
                build_topo(child);

            topo.push_back(tensor);
        }
    };

    //build autodiff graph
    build_topo(shared_from_this());
    if (upstream_grad){
        if (upstream_grad->shape() != shape_)
            throw std::invalid_argument("Gradient shape mismatch!");
        if (device_ == DeviceType::CPU)
            std::copy(upstream_grad->data(), upstream_grad->data() + numel(), grad());
        else 
            cudaMemcpy(grad(), upstream_grad->data(), numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        CudaException::throwIfCUDAErrorsOccurred("Failed load to device in backward!");
    } else {
        if (numel() != 1)
            throw std::runtime_error("Default backward() requires scalar output");
        if (device_ == DeviceType::CPU){
            grad()[0] = 1.0f;
        } else {
            float one = 1.0f;
            cudaMemcpy(grad(), &one, sizeof(float), cudaMemcpyHostToDevice);
            CudaException::throwIfCUDAErrorsOccurred("Failed load to device in backward!");
        }
    }

    //backward pass
    std::reverse(topo.begin(), topo.end());
    for (const auto& tensor : topo){
        if (tensor->backward_fn_){
            tensor->backward_fn_();
        }
    }
    if (device_ == DeviceType::CUDA){
        cudaDeviceSynchronize();
        CudaException::throwIfCUDAErrorsOccurred("Cuda error during backward pass");
    }
}

float& Tensor::operator[](const std::vector<int>& indices){
    size_t idx = linear_index(indices);
    if (device_ == DeviceType::CUDA)
        throw std::invalid_argument("Not implement yet");
    return storage_.data.get()[idx];
}

const float& Tensor::operator[](const std::vector<int>& indices) const{
    size_t idx = linear_index(indices);
    if (device_ == DeviceType::CUDA)
        throw std::invalid_argument("Not implement yet");
    return storage_.data.get()[idx];
}

Tensor::Ptr Tensor::operator*(const Ptr& other) const{
    if (shape_ != other->shape_)
        throw std::invalid_argument("Shape mismatch");
    
    auto result = create(shape_, device_);
    if (device_ == DeviceType::CPU)
        cpu_mul_impl(data(), other->data(), result->data(), numel());
    else
        cuda_mul_impl(data(), other->data(), result->data(), numel());
    
    this->init_grad();
    result->init_grad();
    other->init_grad();
    result->children_ = {get_ptr(), other};
    result->backward_fn_ = [this, other, result](){
        if (device_ == DeviceType::CPU)
            cpu_mul_backward_impl(grad(), other->grad(), result->grad(), data(), other->data(), numel());
        else
            cuda_mul_backward_impl(grad(), other->grad(), result->grad(), data(), other->data(), numel());
    };

    return result;
}

Tensor::Ptr Tensor::operator+(const Ptr& other) const{
    if (shape_ != other->shape_)
        throw std::invalid_argument("Shape mismatch");
    
    auto result = create(shape_, device_);
    if (device_ == DeviceType::CPU)
        cpu_add_impl(data(), other->data(), result->data(), numel());
    else
        cuda_add_impl(data(), other->data(), result->data(), numel());
    
    this->init_grad();
    result->init_grad();
    other->init_grad();
    result->children_ = {get_ptr(), other};
    result->backward_fn_ = [this, other, result](){
        if (device_ == DeviceType::CPU)
            cpu_add_backward_impl(
                grad(), other->grad(), result->grad(), numel()
            );
        else
            cuda_add_backward_impl(
                grad(), other->grad(), result->grad(), numel()
            );
    };

    return result;
}

Tensor::Ptr Tensor::operator/(const Ptr& other) const{
    if (shape_ != other->shape_)
        throw std::invalid_argument("Shape mismatch");
    
    auto result = create(shape_, device_);
    if (device_ == DeviceType::CPU)
        cpu_div_impl(data(), other->data(), result->data(), numel());
    else
        cuda_div_impl(data(), other->data(), result->data(), numel());
    
    this->init_grad();
    result->init_grad();
    other->init_grad();
    result->children_ = {get_ptr(), other};
    result->backward_fn_ = [this, other, result](){
        if (device_ == DeviceType::CPU)
            cpu_div_backward_impl(grad(), other->grad(), result->grad(), data(), other->data(), numel());
        else
            cuda_div_backward_impl(grad(), other->grad(), result->grad(), data(), other->data(), numel());
    };

    return result;
}

Tensor::Ptr Tensor::sum(int dim) const{
    if (dim < 0 || dim >= static_cast<int>(shape_.size()))
        throw std::invalid_argument("Invalid dimension of sum!");
    
    std::vector<int> new_shape;
    for (size_t i = 0; i < shape_.size(); ++i){
        if (i != static_cast<size_t>(dim))
            new_shape.push_back(shape_[i]);
    }
    if (new_shape.empty())
        new_shape.push_back(1);
    auto result = create(new_shape, device_);

    if (device_ == DeviceType::CPU)
        cpu_sum_impl(
            data(), result->data(), shape_.data(),  dim, shape_.size()
        );
    else
        cuda_sum_impl(
            data(), result->data(), shape_.data(),  dim, shape_.size()
        );
    
    this->init_grad();
    result->init_grad();
    result->children_ = {get_ptr()};
    result->backward_fn_ = [this, result, dim](){
        if (device_ == DeviceType::CPU)
            cpu_sum_backward_impl(
                grad(), result->grad(), data(), shape_.data(), dim, shape_.size()
            );
        else
            cuda_sum_backward_impl(
                grad(), result->grad(), data(), shape_.data(), dim, shape_.size()
            );
    };
    return result;
}


