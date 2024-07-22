# PINN para un problema de conducción de calor 2D

Este repositorio contiene el código desarrollado con PyTorch Lightning para un problema de conducción de calor bidimensional en régimen transiente, que consiste en una fuente de calor móvil con distribución Gaussiana sobre una placa plana rectangular de dimensiones $L\times H$. El desarrollo aquí mostrado constituye parte del Trabajo Final de la carrera de Especialización en Inteligencia Artificial (CEIA) del Laboratorio de Sistemas Embebidos (LSE) perteneciente a la Facultad de Ingeniería de la Universidad de Buenos Aires (FIUBA). 


## Parámetros del problema de conducción de calor

Parámetro | Valor |
----------- | ----------- |
Densidad $\rho$ ($\mathrm{kg\,mm^{-3}}$) | $7.6\mathrm{e}{-6}$
Conductividad térmica $k$ ($\mathrm{W\, mm^{-1}\,K^{-1}}$) |$0.025$
Capacidad térmica específica $c_p$ ($\mathrm{J\, kg^{-1}\,K^{-1}}$) | $658$
Difusividad térmica $\alpha$ ($\mathrm{mm}^2\,\mathrm{s}^{-1}$) | $4.80215$
Longitud del dominio $L$ ($\mathrm{mm}$) | $100$
Ancho del dominio $H$ ($\mathrm{mm}$) | $50$
Tiempo total del análisis transiente $t_f$ ($\mathrm{s}$) | $50$
Temperatura inicial $T_0$ (K) | $273$
Potencia total de la fuente Gaussiana $\dot{Q}_T$ ($\mathrm{W}$) | $62.83185$
Velocidad de la fuente $v$ ($\mathrm{mm\, s^{-1}}$) | $2$
Radio característico $r_0$ ($\mathrm{mm}$) | $1.0$ a $10.0$


## Ecuaciones de gobierno

A continuación se escriben las ecuaciones de gobierno del problema de valores iniciales y de borde:

$$
\begin{aligned}
\rho c\frac{\partial T(x,y,t)}{\partial t} &= \nabla\cdot(k\nabla T(x,y,t)) + \dot{Q},\quad \forall(x,y)\in\Omega,\, 0<t\leq t_f,\\
\nabla T(x,y,t)&=0, \quad\partial\Omega,\,0<t\leq t_f,\\
T(x,y,0)&=T_0, \quad \forall(x,y)\in\Omega,\, t=0,
\end{aligned}
$$

donde

$$
\dot{Q}(x,y,t) = \frac{\dot{Q}_T}{\pi r_0^2}e^{-(x-vt)^2/r_0^2}e^{-y^2/r_0^2}
$$


## Adimensionalización

El proceso de adimensionalización de las ecuaciones de gobierno previamente mostradas involucra la definición de las siguientes expresiones:

$$
\xi=\frac{x}{L}, \quad \eta=\frac{y}{L}, \quad \rho_0=\frac{r_0}{L}, \quad \tau=\frac{\alpha}{L^2}t, \quad u=\frac{T}{T_0},
$$

que conducen a las ecuaciones:

$$
\begin{aligned}
\frac{\partial u}{\partial\tau} &= \Delta{u} + \dot{\mathcal{Q}}(\xi,\eta,\tau),\quad\forall(\xi,\eta)\in\Omega^{\ast},\, 0<\tau\leq \tau_f\\
\nabla u(\xi,\eta,\tau)&=0, \quad\partial\Omega^{\ast},\,0<\tau\leq \tau_f,\\
u(\xi,\eta,0)&=u_0, \quad \forall(\xi,\eta)\in\Omega^{\ast},\, \tau=0,
\end{aligned}
$$

donde 

$$
\dot{\mathcal{Q}}(\xi,\eta,\tau) = \frac{\dot{Q}_T^{\ast}}{\pi\rho_0^2}e^{-(\xi-v^{\ast}\tau)^2/\rho_0^2}e^{-\eta^2/\rho_0^2}
$$

y

$$
\dot{Q}_T^{\ast}=\frac{\dot{Q}_T}{k T_0}\quad\text{y}\quad v^{\ast}=\frac{v}{\alpha/L},
$$


## Breve descripción del modelo PINN

El modelo PINN consiste en una red neuronal completamente con entradas $x\in[0,L]$, $y\in[-H/2, H/2]$, $t\in[0, t_f]$ y $r_0\in[1.0; 10.0]$, y salida $\hat{T}(x,y,t,r_0)$. La solución que ofrece es paramétrica dado que permite obtener resultados para diversos radios característicos $r_0$. 