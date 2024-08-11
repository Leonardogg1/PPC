import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

# Função para resolver o sistema mecânico e atualizar os gráficos
def update(val):
    m = slider_m.val
    k = slider_k.val
    V = slider_V.val
    
    # Função para a tensão inicial
    def V0(t):
        return 0.0 if t < 0.5 else V

    def mechanical_system(t, y):
        x, v, i = y
        b = 1
        a = 2.5e-2
        L_prime = 46.8e-3
        L = L_prime * (a / (a + x))
        dL_dx = a * L_prime / (a + x)**2
        F_ext = (i**2 / 2) * dL_dx  # Atualizado com a força externa

        dvdt = (F_ext - b*v - k*(x-0.05)) / m
        dxdt = v

        # Equação elétrica: di/dt = V0 - Ri - i(dL/dx)(dx/dt)
        R = 5
        didt = (V0(t) - R * i - i * dL_dx * v) * ((a + x) / x)

        return [dxdt, dvdt, didt]

    # Condições iniciais
    x0 = 0.05
    v0 = 0.0
    i0 = 0.0
    y0 = [x0, v0, i0]
    T = 4
    dt = 0.0071428571
    t_span = (0, T)

    # Resolver o sistema ODE
    sol = solve_ivp(mechanical_system, t_span, y0, method='RK45', t_eval=np.arange(0, T, dt), dense_output=True)

    # Extrair a solução
    t_mechanical = sol.t
    x_mechanical = sol.y[0]
    v_mechanical = sol.y[1]
    i_mechanical = sol.y[2]

    # Calcular a força magnética
    a = 2.5e-2
    L_prime = 46.8e-3
    dL_dx = a * L_prime / (a + x_mechanical)**2
    force_magnetic = (i_mechanical**2 / 2) * dL_dx

    # Atualizar gráficos
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()

    ax[0].plot(t_mechanical, x_mechanical, label='Displacement (m)')
    ax[0].set_title('Mechanical System SSO')
    ax[0].set_xlabel('Time (s)')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t_mechanical, i_mechanical, label='Current (A)')
    ax[1].set_title('Electrical System SPO')
    ax[1].set_xlabel('Time (s)')
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t_mechanical, force_magnetic, label='Magnetic Force (N)')
    ax[2].set_title('Magnetic Force')
    ax[2].set_xlabel('Time (s)')
    ax[2].legend()
    ax[2].grid()

    plt.draw()

# Criar a figura e os eixos para os gráficos
fig, ax = plt.subplots(3, 1, figsize=(20, 10))
plt.subplots_adjust(left=0.15, right=0.85, hspace=0.5)

# Criar sliders
ax_slider_m = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_k = plt.axes([0.15, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_V = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_m = Slider(ax_slider_m, 'Mass (M)', 0.1, 1.0, valinit=0.5)
slider_k = Slider(ax_slider_k, 'Spring Constant (K)', 0.001, 300, valinit=60)
slider_V = Slider(ax_slider_V, 'Initial Voltage (V)', 0, 200, valinit=10)

# Conectar os sliders à função de atualização
slider_m.on_changed(update)
slider_k.on_changed(update)
slider_V.on_changed(update)

# Plotar os gráficos iniciais
update(None)

plt.show()
