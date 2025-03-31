from nanograd import DeviceType, tensor
import numpy as np

class TestCreation:
    def test_create_shape_cpu(self):
        a = tensor.create_shape([1, 2, 5], DeviceType.CPU)
        assert a.shape() == [1, 2, 5]
        assert a.device() == DeviceType.CPU
        assert all(a.data() == np.array([0] * 10))
        assert a.grad().size == 0

    def test_create_shape_cuda(self):
        a = tensor.create_shape(shape=[1, 2, 5], device=DeviceType.CUDA)
        assert a.shape() == [1, 2, 5]
        assert a.device() == DeviceType.CUDA
    
    def test_create_data_cpu(self):
        a = tensor.create_data([5] * 100, [5, 2, 10], DeviceType.CPU)
        assert a.shape() == [5, 2, 10]
        assert a.device() == DeviceType.CPU
        assert all(a.data() == np.array([5] * 100))
        assert a.grad().size == 0
    
    def test_create_data_cuda(self):
        a = tensor.create_data([5] * 100, [5, 2, 10], DeviceType.CUDA)
        assert a.shape() == [5, 2, 10]
        assert a.device() == DeviceType.CUDA


class TestInterface:
    def test_shape(self):
        shape = [5, 2, 10]
        shape2 = []
        a = tensor.create_shape(shape=shape, device=DeviceType.CPU)
        b = tensor.create_shape(shape=shape, device=DeviceType.CUDA)
        c = tensor.create_shape(shape=shape2, device=DeviceType.CPU)
        d = tensor.create_shape(shape=shape2, device=DeviceType.CUDA)
        assert a.shape() == shape
        assert b.shape() == shape
        assert c.shape() == shape2
        assert d.shape() == shape2
    
    def test_device(self):
        shape = []
        a = tensor.create_shape(shape=shape, device=DeviceType.CPU)
        b = tensor.create_shape(shape=shape, device=DeviceType.CUDA)
        assert a.device() == DeviceType.CPU
        assert b.device() == DeviceType.CUDA
    
    def test_numel(self):
        shape = []
        a = tensor.create_shape(shape, device=DeviceType.CPU)
        b = tensor.create_shape(shape, device=DeviceType.CUDA)
        shape2 = [10, 2, 5]
        c = tensor.create_shape(shape=shape2, device=DeviceType.CPU)
        d = tensor.create_shape(shape=shape2, device=DeviceType.CUDA)
        assert a.numel() == 0
        assert b.numel() == 0
        assert c.numel() == 100
        assert d.numel() == 100
    
    def test_to(self):
        shape = []
        shape2 = [10, 2, 5]
        a = tensor.create_shape(shape, device=DeviceType.CPU)
        b = tensor.create_shape(shape=shape2, device=DeviceType.CUDA)
        c = a.to(DeviceType.CUDA)
        d = b.to(DeviceType.CPU)
        assert a.device() == DeviceType.CPU
        assert b.device() == DeviceType.CUDA
        assert c.device() == DeviceType.CUDA
        assert d.device() == DeviceType.CPU
    
    def test_zero_grad(self):
        shape = [10, 2, 5]
        a = tensor.create_shape(shape, device=DeviceType.CPU)
        b = tensor.create_shape(shape, device=DeviceType.CPU)
        assert a.grad().size == 0
        assert b.grad().size == 0
        a + b
        assert a.grad().size == 100
        assert b.grad().size == 100
        a.zero_grad()
        b.zero_grad()
        assert all(a.grad() == [0] * 100)
        assert all(b.grad() == [0] * 100)
    

class TestOperations:
    def test_add_cpu(self):
        shape = [5, 2, 10]
        a = tensor.create_data([5] * 100, shape, DeviceType.CPU)
        b = tensor.create_data([3] * 100, shape, DeviceType.CPU)
        c = a + b
        assert all(c.data() == [8]*100)
        output_grad = tensor.create_data([1] * 100, shape, DeviceType.CPU)
        c.backward(output_grad)
        assert all(c.grad() == [1]*100)
        assert all(a.grad() == [1]*100)
        assert all(b.grad() == [1]*100)

    def test_add_cuda(self):
        shape = [5, 2, 10]
        a = tensor.create_data([5] * 100, shape, DeviceType.CUDA)
        b = tensor.create_data([3] * 100, shape, DeviceType.CUDA)
        c = a + b
        output_grad = tensor.create_data([1] * 100, shape, DeviceType.CUDA)
        c.backward(output_grad)
        a_cpu = a.to(DeviceType.CPU)
        b_cpu = b.to(DeviceType.CPU)
        c_cpu = c.to(DeviceType.CPU)
        assert all(c_cpu.data() == [8]*100)
        assert all(a_cpu.grad() == [1]*100)
        assert all(b_cpu.grad() == [1]*100)
        assert all(c_cpu.grad() == [1]*100)

    def test_add_double_backward(self):
        shape = [5, 2, 10]
        a = tensor.create_data([5] * 100, shape, DeviceType.CPU)
        b = tensor.create_data([3] * 100, shape, DeviceType.CPU)
        c = a + b
        assert all(c.data() == [8]*100)
        output_grad = tensor.create_data([1] * 100, shape, DeviceType.CPU)
        c.backward(output_grad)
        c.backward(output_grad)
        assert all(c.grad() == [1]*100)
        assert all(a.grad() == [2]*100)
        assert all(b.grad() == [2]*100)
    
    def test_mul_cpu(self):
        shape = [5, 2, 10]
        a = tensor.create_data([5] * 100, shape, DeviceType.CPU)
        b = tensor.create_data([3] * 100, shape, DeviceType.CPU)
        c = a * b
        assert all(c.data() == [15]*100)
        output_grad = tensor.create_data([1] * 100, shape, DeviceType.CPU)
        c.backward(output_grad)
        assert all(c.grad() == [1]*100)
        assert all(a.grad() == [3]*100)
        assert all(b.grad() == [5]*100)

    def test_mul_cuda(self):
        shape = [5, 2, 10]
        a = tensor.create_data([5] * 100, shape, DeviceType.CUDA)
        b = tensor.create_data([3] * 100, shape, DeviceType.CUDA)
        c = a * b
        output_grad = tensor.create_data([1] * 100, shape, DeviceType.CUDA)
        c.backward(output_grad)
        a_cpu = a.to(DeviceType.CPU)
        b_cpu = b.to(DeviceType.CPU)
        c_cpu = c.to(DeviceType.CPU)
        assert all(c_cpu.data() == [15]*100)
        assert all(c_cpu.grad() == [1]*100)
        assert all(a_cpu.grad() == [3]*100)
        assert all(b_cpu.grad() == [5]*100)

    def test_mul_double_backward(self):
        shape = [5, 2, 10]
        a = tensor.create_data([5] * 100, shape, DeviceType.CPU)
        b = tensor.create_data([3] * 100, shape, DeviceType.CPU)
        c = a * b
        assert all(c.data() == [15]*100)
        output_grad = tensor.create_data([1] * 100, shape, DeviceType.CPU)
        c.backward(output_grad)
        c.backward(output_grad)
        assert all(c.grad() == [1]*100)
        assert all(a.grad() == [6]*100)
        assert all(b.grad() == [10]*100)

    def test_div_cpu(self):
        shape = [5, 2, 10]
        a = tensor.create_data([6] * 100, shape, DeviceType.CPU)
        b = tensor.create_data([3] * 100, shape, DeviceType.CPU)
        c = a / b
        assert all(c.data() == [2]*100)
        output_grad = tensor.create_data([1] * 100, shape, DeviceType.CPU)
        c.backward(output_grad)
        assert all(c.grad() == [1]*100)
        assert np.allclose(a.grad(), np.array([1/3]*100))
        assert np.allclose(b.grad(), -np.array([6/9]*100))

    def test_div_cuda(self):
        shape = [5, 2, 10]
        a = tensor.create_data([6] * 100, shape, DeviceType.CUDA)
        b = tensor.create_data([3] * 100, shape, DeviceType.CUDA)
        c = a / b
        output_grad = tensor.create_data([1] * 100, shape, DeviceType.CUDA)
        c.backward(output_grad)
        c_cpu = c.to(DeviceType.CPU)
        a_cpu = a.to(DeviceType.CPU)
        b_cpu = b.to(DeviceType.CPU)
        assert all(c_cpu.data() == [2]*100)
        assert all(c_cpu.grad() == [1]*100)
        assert np.allclose(a_cpu.grad(), np.array([1/3]*100))
        assert np.allclose(b_cpu.grad(), -np.array([6/9]*100))

    def test_div_double_backward(self):
        shape = [5, 2, 10]
        a = tensor.create_data([6] * 100, shape, DeviceType.CPU)
        b = tensor.create_data([3] * 100, shape, DeviceType.CPU)
        c = a / b
        assert all(c.data() == [2]*100)
        output_grad = tensor.create_data([1] * 100, shape, DeviceType.CPU)
        c.backward(output_grad)
        c.backward(output_grad)
        assert all(c.grad() == [1]*100)
        assert np.allclose(a.grad(), np.array([2/3]*100))
        assert np.allclose(b.grad(), -np.array([12/9]*100))
    
    def test_sum_cpu(self):
        shape = [5, 2, 10]
        a = tensor.create_data([1] * 100, shape, DeviceType.CPU)
        b = a.sum(0)
        assert all(b.shape() == np.array([2, 10]))
        assert np.allclose(b.data(), np.array([5] * 20))
        output_grad_b = tensor.create_data([1] * 20, [2, 10], DeviceType.CPU)
        b.backward(output_grad_b)
        assert np.allclose(a.grad(), np.array([1] * 100))
        a.zero_grad()

        c = a.sum(1)
        assert all(c.shape() == np.array([5, 10]))
        assert np.allclose(c.data(), np.array([2] * 50))
        output_grad_c = tensor.create_data([1] * 50, [5, 10], DeviceType.CPU)
        c.backward(output_grad_c)
        assert np.allclose(a.grad(), np.array([1] * 100))
        a.zero_grad()

        d = a.sum(2)
        assert all(d.shape() == np.array([5, 2]))
        assert all(d.grad() == np.array([0] * 10))
        assert np.allclose(d.data(), np.array([10] * 10))
        output_grad_d = tensor.create_data([1] * 10, [5, 2], DeviceType.CPU)
        d.backward(output_grad_d)
        assert np.allclose(a.grad(), np.array([1] * 100))
        a.zero_grad()

    def test_sum_cuda(self):
        shape = [5, 2, 10]
        a = tensor.create_data([1] * 100, shape, DeviceType.CUDA)
        b = a.sum(0)
        output_grad_b = tensor.create_data([1] * 20, [2, 10], DeviceType.CUDA)
        b.backward(output_grad_b)
        b_cpu = b.to(DeviceType.CPU)
        a_cpu = a.to(DeviceType.CPU)
        assert all(b_cpu.shape() == np.array([2, 10]))
        assert np.allclose(b_cpu.data(), np.array([5] * 20))
        assert np.allclose(a_cpu.grad(), np.array([1] * 100))
        a.zero_grad()

        c = a.sum(1)
        output_grad_c = tensor.create_data([1] * 50, [5, 10], DeviceType.CUDA)
        c.backward(output_grad_c)
        c_cpu = c.to(DeviceType.CPU)
        a_cpu = a.to(DeviceType.CPU)
        assert all(c_cpu.shape() == np.array([5, 10]))
        assert np.allclose(c_cpu.data(), np.array([2] * 50))
        assert np.allclose(a_cpu.grad(), np.array([1] * 100))
        a.zero_grad()

        d = a.sum(2)
        output_grad_d = tensor.create_data([1] * 10, [5, 2], DeviceType.CUDA)
        d.backward(output_grad_d)
        d_cpu = d.to(DeviceType.CPU)
        a_cpu = a.to(DeviceType.CPU)
        assert all(d_cpu.shape() == np.array([5, 2]))
        assert np.allclose(d_cpu.data(), np.array([10] * 10))
        assert np.allclose(a_cpu.grad(), np.array([1] * 100))
    
    def test_mixed_backward_cpu(self):
        shape = [5, 2, 10]
        a = tensor.create_data([2] * 100, shape, DeviceType.CPU)
        b = tensor.create_data([3] * 100, shape, DeviceType.CPU)
        c = tensor.create_data([5] * 100, shape, DeviceType.CPU)
        d = (a + b) * c
        assert all(d.data() == [25]*100)
        output_grad = tensor.create_data([1] * 100, shape, DeviceType.CPU)
        d.backward(output_grad)
        assert all(d.grad() == [1]*100)
        assert np.allclose(a.grad(), np.array([5]*100))
        assert np.allclose(b.grad(), np.array([5]*100))

    def test_mixed_backward_cuda(self):
        shape = [5, 2, 10]
        a = tensor.create_data([2] * 100, shape, DeviceType.CUDA)
        b = tensor.create_data([3] * 100, shape, DeviceType.CUDA)
        c = tensor.create_data([5] * 100, shape, DeviceType.CUDA)
        d = (a + b) * c
        output_grad = tensor.create_data([1] * 100, shape, DeviceType.CUDA)
        d.backward(output_grad)
        d_cpu = d.to(DeviceType.CPU)
        a_cpu = a.to(DeviceType.CPU)
        b_cpu = b.to(DeviceType.CPU)
        assert all(d_cpu.data() == [25]*100)
        assert all(d_cpu.grad() == [1]*100)
        assert np.allclose(a_cpu.grad(), np.array([5]*100))
        assert np.allclose(b_cpu.grad(), np.array([5]*100))
