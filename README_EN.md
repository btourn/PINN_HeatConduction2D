# PINN for a 2D heat conduction problem

This repository contains the scripts written using PyTorch Lightning for a transient, two-dimensional heat conduction problem that involves a Gaussian moving heat source over a $l\times H$ rectangular plane plate. This work is part of the Artificial Intelligence Posgraduate Specialization program final project from the Embebebed Systems Laboratory (LSE) of the Engineering Faculty of the University of Buenos Aires (UBA).


## Parameters of the heat conduction problema

Parameter | Value |
----------- | ----------- |
Density $\rho$ ($\mathrm{kg\,mm^{-3}}$) | $7.6\mathrm{e}{-6}$
Thermal conductivity $k$ ($\mathrm{W\, mm^{-1}\,K^{-1}}$) |$0.025$
Specific heat capacity $c\\_p$ ($\mathrm{J\, kg^{-1}\,K^{-1}}$) | $658$
Thermal diffusivity $\alpha$ ($\mathrm{mm}^2\,\mathrm{s}^{-1}$) | $4.80215$
Domain length $L$ ($\mathrm{mm}$) | $100$
Domain width $H$ ($\mathrm{mm}$) | $50$
Transient analysis final time $t\_f$ ($\mathrm{s}$) | $50$
Initial temperature $T\_0$ (K) | $273$
Total power of the heat source $\dot{Q}\_T$ ($\mathrm{W}$) | $62.83185$
Heat source velocity $v$ ($\mathrm{mm\, s^{-1}}$) | $2$
Characteristic radius $r\_0$ ($\mathrm{mm}$) | $1.0$ a $10.0$


## Governing equations

The following are the governing equations of the initial and boundary value problem:

$$ 
\begin{aligned}
\rho c\_p\frac{\partial T(x,y,t)}{\partial t}&=\nabla\cdot(k\nabla T(x,y,t)) + \dot{Q}\, \quad \forall(x,y)\in\Omega\, \quad 0\lt t\leq t\_f,\\
\nabla T(x,y,t)&=0,\quad\partial\Omega,\quad 0\lt t\leq t\_f,\\
T(x,y,0)&=T\_0,\quad\forall(x,y)\in\Omega,\quad t=0,
\end{aligned} 
$$

where

$$\dot{Q}(x,y,t) = \frac{\dot{Q}\_T}{\pi r\_0^2}e^{-(x-vt)^2/r\_0^2}e^{-y^2/r\_0^2}.$$


## Nondimensionalization

The nondimensionalization process of the previosuly shown governing equations involves the definition of the following expressions:

$$
\xi=\frac{x}{L}, \quad \eta=\frac{y}{L}, \quad \rho\_0=\frac{r\_0}{L}, \quad \tau=\frac{\alpha}{L^2}t, \quad u=\frac{T}{T\_0},
$$

that lead to the following equations:

$$
\begin{aligned}
\frac{\partial u}{\partial\tau} &= \Delta{u} + \dot{\mathcal{Q}}(\xi,\eta,\tau),\quad\forall(\xi,\eta)\in\Omega^{\ast}, \quad 0<\tau\leq \tau\_f\\
\nabla u(\xi,\eta,\tau)&=0, \quad\partial\Omega^{\ast}, \quad 0<\tau\leq \tau\_f,\\
u(\xi,\eta,0)&=u\_0, \quad \forall(\xi,\eta)\in\Omega^{\ast}, \quad \tau=0,
\end{aligned}
$$

where 

$$
\dot{\mathcal{Q}}(\xi,\eta,\tau) = \frac{\dot{Q}\_T^{\ast}}{\pi\rho\_0^2}e^{-(\xi-v^{\ast}\tau)^2/\rho\_0^2}e^{-\eta^2/\rho\_0^2}
$$

and

$$
\dot{Q}\_T^{\ast}=\frac{\dot{Q}\_T}{k T\_0}\quad\mathrm{y}\quad v^{\ast}=\frac{v}{\alpha/L},
$$


## Brief description of the PINN model

The PINN model consists in a 4-features-inputs and 1-output fully connected neural network. The inputs are $x\in[0,L]$, $y\in[-H/2, H/2]$, $t\in[0, t\_f]$ y $r\_0\in[1.0; 10.0]$. The output is the temperature $\hat{T}(x,y,t,r\_0)$. The model offers a parametric solution since it allows obtaining results for any value of the characteristic radius $r_0$ within its range.

The model was trained under an unsupervised learning approach (i.e., without labelled data) using a LBFGS algorithm. The loss function is defined as

$$
\mathcal{L} = \mathcal{L}\_{\mathrm{PDE}} + \mathcal{L}\_{\mathrm{BC}} + \mathcal{L}\_{\mathrm{IC}}
$$

where

$$
\begin{aligned}
\mathcal{L}\_{\mathrm{PDE}} &= \frac{1}{N\_{\mathrm{PDE}}}\sum\_{N\_{\mathrm{PDE}}}\Bigg[\rho c\_p\frac{\partial T(x,y,t)}{\partial t} - \nabla\cdot(k\nabla T(x,y,t)) - \dot{Q}\Bigg]^2\\
\mathcal{L}\_{\mathrm{BC}} &= \frac{1}{N\_{\mathrm{BC}}}\sum\_{N\_{\mathrm{BC}}}\big[\nabla T(x,y,t)\big]^2\\
\mathcal{L}\_{\mathrm{IC}} &= \frac{1}{N\_{\mathrm{IC}}}\sum\_{N\_{\mathrm{IC}}}\big[T(x,y,0) - T\_0\big]^2
\end{aligned}
$$
