<a name="readme-top"></a>



<!-- PROJECT LOGO -->
<div align="center">


  <h2 align="center">
    Auto Gradient
  </h2>
</div>




<div>
<div align="center">
    <img src="images/graph.png" alt="Logo" width="1000">
</div>

<br/>
<br/>

Library for auto generating derivatives on-the-fly
<br/>
All basic math operators are supported (add, mult, exponentiate, etc.)
<br/>
value.backward() will calculate the gradients and save to each value in the lineage
<br/>

```python
from autograd import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```
<!-- GETTING STARTED -->

## Usage

Clone the repo
   ```sh
   git clone https://github.com/ryansereno/micrograd
   ```
Import
   ```python
    from autograd import Value
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>










