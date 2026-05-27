# 参考文献

## PINN 基础

[1] M. Raissi, P. Perdikaris, G.E. Karniadakis. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378: 686–707, 2019.

[2] G.E. Karniadakis, I.G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, L. Yang. *Physics-informed machine learning*. Nature Reviews Physics, 3(6): 422–440, 2021.

## 结构化网格 PINN / 有限差分 PINN

[3] C. Wang, S. Li, D. He, H. Wang, P. Perdikaris. *Is L2 physics-informed loss always good for training?* arXiv:2112.10331, 2021.

[4] S. Wang, Y. Teng, P. Perdikaris. *Understanding and mitigating gradient flow pathologies in physics-informed neural networks*. SIAM Journal on Scientific Computing, 43(5): A3055–A3081, 2021.

[5] S. Wang, X. Yu, P. Perdikaris. *When and why PINNs fail to train: A neural tangent kernel perspective*. Journal of Computational Physics, 449: 110768, 2022.

## GPU 算子融合 / Triton

[6] P. Tillet, H.T. Kung, D. Cox. *Triton: an intermediate language and compiler for tiled neural network computations*. PLDI 2019. arXiv:1903.01216

[7] J. Ansel, E. Yang, H. He, N. Gimelshein, A. Jain, et al. *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation*. ASPLOS 2024.

## 符号回归与物理发现

[8] S.L. Brunton, J.L. Proctor, J.N. Kutz. *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*. PNAS, 113(15): 3932–3937, 2016.

[9] M. Cranmer. *Interpretable machine learning for science with PySR and SymbolicRegression.jl*. arXiv:2305.01582, 2023.

[10] M. Cranmer, A. Sanchez-Gonzalez, P. Battaglia, R. Xu, K. Cranmer, D. Spergel, S. Ho. *Discovering symbolic models from deep learning with inductive biases*. NeurIPS 2020.

[11] S.M. Udrescu, M. Tegmark. *AI Feynman: A physics-inspired method for symbolic regression*. Science Advances, 6(16): eaay2631, 2020.

## Guderley 收缩激波

[12] G. Guderley. *Starke kugelige und zylindrische Verdichtungsstöße in der Nähe des Mittelpunktes*. Luftfahrtforschung, 19(9): 302–312, 1942.

[13] L.I. Sedov. *Similarity and Dimensional Methods in Mechanics*. Academic Press, 1959.

[14] R.E. Kidder. *Theory of homogeneous implosion of thermonuclear fuel*. Nuclear Fusion, 1974.

## Taylor-Maccoll 锥面流

[15] G.I. Taylor, J.W. Maccoll. *The air pressure on a cone moving at high speeds*. Proceedings of the Royal Society of London A, 139(838): 278–311, 1933.

## Neural Operator（背景/对比）

[16] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar. *Fourier Neural Operator for Parametric Partial Differential Equations*. ICLR 2021. arXiv:2010.08895
