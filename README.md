# Artificial Neural Networks for the deconvolution of the Abel Transform (ANNAbel)
In this repository we present a simple pyTorch implementation of ANNABEL for the deconvolution of soot volume fraction and temperature of 2D measurements.

The Abel transform is defined as follows:

$$F(y,z)=2\int _{y}^{\infty }{\frac {f(r,z)r\,dr}{\sqrt {r^{2}-y^{2}}}}$$

The core idea of this approach consist on aproximate $f$ with an artificial neural network, such that $f(r,z) \approx \tilde f(r,z) = \text{NN}(r,z; \theta)$. And then minimize the following loss function:

$$\mathcal{L} = \frac{1}{N_r N_z}\sum_k^{N_z}\sum_i^{N_r} (F(y_i, z_k)-\tilde F(y_i, z_k))^2$$

where $\tilde F(y_i, z)$ corresponds to the discretization of the integral:

$$\tilde F(y_i, z) = \Delta r \sum_{j=i}^\infty W_{ij} \tilde f(r_j, z)$$

Where:

$$
W_{ij} =
\begin{cases}
    0, & j < i, \\
    \sqrt{(2j + 1)^2 - 4i^2}, & j = i, \\
    \sqrt{(2j + 1)^2 - 4i^2} - \sqrt{(2j - 1)^2 - 4i^2}, & j > i.
\end{cases}
$$

This discretization is what is know as Onion Peeling.

# Cite the original article

```bibtex
@article{ESCUDERO2024105493,
title = {Robust automatic retrieval of soot volume fraction, temperature and radiation for axisymmetric flames},
journal = {Proceedings of the Combustion Institute},
volume = {40},
number = {1},
pages = {105493},
year = {2024},
issn = {1540-7489},
doi = {https://doi.org/10.1016/j.proci.2024.105493},
url = {https://www.sciencedirect.com/science/article/pii/S1540748924003018},
author = {Felipe Escudero and Victor Chernov and Juan J. Cruz and Efraín Magaña and Benjamín Herrmann and Andrés Fuentes},
keywords = {Forward projection, Spatially-resolved measurements, Light extinction/emission, Soot, Artificial neural network},
abstract = {This work presents a robust methodology to retrieve local soot properties from line-of-sight integrated measurements without the need to invert a poorly-conditioned matrix arising from the flame geometry and discretization procedureFirst, a forward fit method is presented. Another method, utilizing an Artificial Neural Network informed by the Abel equation (ANNAbel), is then introduced to circumvent the drawbacks of the forward fit method. Both methods are capable to retrieve soot volume fraction, temperature and radiation satisfactorily from experimental data of an ethylene coflow non-premixed flame, without the need for a tuning a regularization parameter. The ANNAbel approach exhibited greater smoothness for retrieved properties, with lower errors when comparing the reconstructed data against the original experimental data. This was also evident when comparing local soot properties in a numerical framework. The ANNAbel approach also showed high resilience to increased levels of noise, contrary to the fitting approach and classical deconvolution methods. Finally, the ANNAbel method was capable to obtain the local properties even with simulated corrupted data, with a level of precision slightly lower than treating the original experimental data. On the contrary, the rest of the methods failed to perform this task. The ANNAbel method is then a promising approach for the robust and accurate determination of local flame properties, which is especially important for obtaining complex soot properties such as size and composition, where involved data treatment is required, and the results are sensitive to noise.}
}
```
