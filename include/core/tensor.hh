#pragma once
#include <cstdlib>
#include <memory>
#include <functional>
#include <vector>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

enum class DeviceType{CPU, CUDA};

class Tensor : public std::enable_shared_from_this<Tensor>{
    using Ptr = std::shared_ptr<Tensor>;

    private:
        DeviceType device_;
        size_t offset_ = 0;
        std::vector<int> shape_;
        std::vector<int> stride_;
        std::vector<Ptr> children_;
        std::function<void()> backward_fn_;
        mutable bool grad_initialized_ = false;

        struct DataHandle{
            std::shared_ptr<float> data;
            mutable std::shared_ptr<float> grad;
            size_t size;
        };
        DataHandle storage_;

        Tensor() = delete;
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;
        Tensor(Tensor&&) = default;
        Tensor& operator=(Tensor&&) = default;

        //data handling
        void init_grad() const;
        void allocate_storage();
        void calculate_strirde();
        size_t linear_index(const std::vector<int>& indices) const;

     public:
        Tensor(
            DeviceType device, const std::vector<int>& shape, std::shared_ptr<float> data, std::shared_ptr<float> grad
        );
        ~Tensor();
        static Ptr create(
            const std::vector<int>& shape, DeviceType device = DeviceType::CPU
        );
        static Ptr create(
            const std::vector<float>& data, const std::vector<int> &shape, DeviceType device = DeviceType::CPU
        );
        Tensor::Ptr get_ptr(){
            return std::static_pointer_cast<Tensor>(shared_from_this());
        }
        Tensor::Ptr get_ptr() const{
            return std::const_pointer_cast<Tensor>(shared_from_this());
        }

        //interface
        DeviceType device() const noexcept {return device_;}
        const std::vector<int>& shape() const noexcept {return shape_;}
        float* data() const noexcept {return storage_.data.get() + offset_;}
        float* grad() const {return storage_.grad.get() + offset_;}
        size_t numel() const noexcept;
        Ptr to(DeviceType target_device);
        Ptr sum(int dim) const;
        void zero_grad();
        void backward(const Tensor::Ptr& upstream_grad = nullptr);

        Ptr operator+(const Ptr& other) const;
        Ptr operator*(const Ptr& other) const;
        Ptr operator/(const Ptr& other) const;
        float& operator[](const std::vector<int>& indices);
        const float& operator[](const std::vector<int>& indices) const;

        //Future
        // Tensor view(const std::vector<int>& new_shape) const;
        // Ptr transpose(int dim1, int dim2) const;
        // Tensor contiguous() const;
};
