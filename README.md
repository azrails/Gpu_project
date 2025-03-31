# nanograd (v 0.0.1a)
Репозиторий является учебным проектом и подходит только в целях обучения (покрайней мере пока).

### Задача
Репозиторий является вольной реализацией autograd движка, написанный на языке C++/СUDA - core, с использованием pybind11 для привязки к python.
Для сборки предпочтительным является пакетный менеджер [uv](https://github.com/astral-sh/uv), второй вариант через cmake после чего появится импортируемая shared библиотека.
```bash
mkdir build && cd build
cmake ..
make
```

Базовой единицей движка является класс tensor поддерживаюший хранение элементов и сборку графа вычислений и класс DeviceType, предназначенный для манипулирования данными тензоров.

(За большим числом примеров смотри tests, notebooks)

### examples
1) Создание пустого тензора

```py
from nanograd import tensor, DeviceType

#CPU tensor ~ torch.zeros((shape), device='cpu')
a = tensor.create_shape(shape=[1, 2, 5], device=DeviceType.CPU)

#CUDA tensor ~ torch.zeros((shape), device='cuda')
b = tensor.create_shape(shape=[1, 2, 5], device=DeviceType.CUDA)
```

2) Создание тензора с данными (на данный момент принимает данные только в виде одномерного массива с заданными размерами)
```py
from nanograd import tensor, DeviceType

#CPU tensor ~ torch.tensor((shape), device='cpu', dtype=torch.float32)
a = tensor.create_shape(list(range(3 * 2 * 5)), shape=[3, 2, 5], device=DeviceType.CPU)

#CUDA tensor ~ torch.zeros((shape), device='cuda', dtype=torch.float32)
b = tensor.create_shape(list(range(3 * 2 * 5)), shape=[3, 2, 5], device=DeviceType.CUDA)
```

3) Поэлементные операции (поддерживаемые на текущий момент)
```py
from nanograd import tensor, DeviceType

a = tensor.create_shape(list(range(3 * 2 * 5)), [3, 2, 5], device=DeviceType.CUDA)
b = tensor.create_shape(list(range(3 * 2 * 5)), [3, 2, 5], device=DeviceType.CUDA)
c = a * b
d = a + b
e = a / b
```

4) Перенос на девайс / с девайса
```py
from nanograd import tensor, DeviceType

a = tensor.create_shape(list(range(3 * 2 * 5)), [3, 2, 5], device=DeviceType.CUDA)
a_cpu = a.to(DeviceType.CPU)
a_cuda = a_cpu.to(DeviceType.CUDA)
```

5) Вспомогательные функции
```py
from nanograd import tensor, DeviceType

a = tensor.create_shape(list(range(3 * 2 * 5)), [3, 2, 5], device=DeviceType.CUDA)
a.numel() # -> int: 30
a.device() # -> DeviceType: CUDA
a.shape() # -> list[int]: [3, 2, 5]
a.data() # -> np.array[float]
b.grad() # -> np.array[float]
```
Замечание: пока не реализованно автоматическое переклюяение с cuda на cpu, могут возникнуть проблемы при попытках манипулирования данными.
(в случае cpu все ок)

6) Дополнительные операции
```py
from nanograd import tensor, DeviceType

a = tensor.create_shape([1, 2, 3, 4, 5, 6], [2, 3])
b = a.sum(0)
b.shape() # -> [3]
c = a.sum(1)
c.shape() # -> [2]
```

7) Автоматическое дифференцирование
```py
from nanograd import tensor, DeviceType

a = tensor.create_shape([1, 2, 3, 4], [2, 2])
b = tensor.create_shape([1, 2, 3, 4], [2, 2])
c = tensor.create_shape([2, 2, 2, 2], [2, 2])

#В этом моменте данные автоматически связываются с графом вычислений
#Так же переприсваивание допустимо, вычисления не потеряются
c = (a + b) / c

#Просчет градиентов
с = c.sum(0).sum(0) # -> int
c.backward()
#Очистка градиента (только у конкретного тензора)
c.zero_grad()

#В случае нескальярного выхода необходимо передать начальные градиенты (смотри torch.autograd)
upstream_grad = tensor(np.ones(5), [5])
c.backward(upstream_grad)
```
### todo
Проект пока еще очень сырой и был обнаружен значительный просчет в архитектуре.

### In progress:
- [ ] Небольшая смена архитектуры хранения данных
- [ ] Необходимые функции 
  - [ ] cos/sin
  - [ ] mat mul
  - [ ] batch mat mul
  - [ ] view
  - [ ] contiguous
  - [ ] transpose
- [ ] Оптимизаторы
  - [ ] AdamW fused
  - [ ] RmsProp
  - [ ] SGD
- [ ] Базовые слои
  - [ ] Linear
  - [ ] Conv
  - [ ] Scaled dot product attention
- [ ] Дистрибутивная комуникация
 - [ ] DDP

### ✓
- [x] Базовый api хранилиша данных (cpu/cuda)
- [x] Поэлементные функции (cpu/cuda)
- [x] Базовый граф автодифференцирования (cpu/cuda)
