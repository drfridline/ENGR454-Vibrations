# viblib.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
from scipy import fft
from scipy import integrate

def free_response(m=10, c=1, k=100, x0=1, v0=-1, max_time=10):
    r"""Free response of a second order linear oscillator.

    Returns t, x, v, zeta, omega, omega_d and A resulting from the
    free response of a second order linear ordinary differential
    equation defined by
    :math:`m\ddot{x} + c \dot{x} + k x = 0`
    given initial conditions :math:`x_0` and :math:`\dot{x}_0 = v_0` for
    :math:`0 < t < t_{max}`

    Parameters
    ----------
    m, c, k :  floats, optional
        mass, damping coefficient, stiffness
    x0, v0:  floats, optional
        initial displacement, initial velocity
    max_time: float, optional
        end time for :math:`x(t)`

    Returns
    -------
    t, x, v : ndarrays
        time, displacement, and velocity
    zeta, omega, omega_d, A : floats
        damping ratio, undamped natural frequency, damped natural frequency,
        Amplitude

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    
    >>> t, x, *_ = vtb.free_response() # *_ ignores all other returns
    >>> plt.plot(t,x)
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.xlabel('Time (sec)')
    Text(0.5, 0, 'Time (sec)')
    >>> plt.ylabel('Displacement (m)')
    Text(0, 0.5, 'Displacement (m)')
    >>> plt.title('Displacement versus time')
    Text(0.5, 1.0, 'Displacement versus time')
    >>> plt.grid(True)

    """
    omega = np.sqrt(k / m)
    zeta = c / 2 / omega / m
    omega_d = omega * np.sqrt(1 - zeta ** 2)
    A = np.sqrt(x0 ** 2 + (v0 + omega * zeta * x0) ** 2 / omega_d ** 2)

    def sdofs_deriv(x_xd, t0, m=m, c=c, k=k):
        x, xd = x_xd
        return [xd, -c / m * xd - k / m * x]

    z0 = np.array([[x0, v0]])
    # Solve for the trajectories
    t = np.linspace(0, max_time, int(250 * max_time))
    z_t = np.asarray([integrate.odeint(sdofs_deriv, z0i, t)
                      for z0i in z0])

    x, y = z_t[:, :].T
    return t, x, y, zeta, omega, omega_d, A

def time_plot(t, x, y, zeta, omega, omega_d, A):
    fig = plt.figure()
    fig.suptitle('Displacement vs Time')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time')
    ax.set_ylabel('Displacement')
    ax.grid(True)
    ax.plot(t, x)
    if zeta < 1:
        ax.plot(t, A * np.exp(-zeta * omega * t), '--C2',
                linewidth=1)
        ax.plot(t, -A * np.exp(-zeta * omega * t), '--C2',
                linewidth=1, label=r'$A e^{- \zeta \omega t}$')
        tmin, tmax, xmin, xmax = ax.axis()
        ax.text(.60 * tmax, .80 * (xmax - xmin) + xmin,
                r'$\omega$ = %0.4f rad/sec' % (omega))
        ax.text(.60 * tmax, .75 * (xmax - xmin)
                + xmin, r'$\zeta$ = %0.4f' % (zeta))
        ax.text(.60 * tmax, .70 * (xmax - xmin) + xmin,
                r'$\omega_d$ = %0.4f rad/sec' % (omega_d))
    else:
        tmin, tmax, xmin, xmax = ax.axis()
        ax.text(.60 * tmax, .80 * (xmax - xmin)
                + xmin, r'$\zeta$ = %0.4f' % (zeta))
        ax.text(.60 * tmax, .75 * (xmax - xmin) + xmin,
                r'$\lambda_1$ = %0.4f' %
                (zeta * omega - omega * (zeta ** 2 - 1)))
        ax.text(.60 * tmax, .70 * (xmax - xmin) + xmin,
                r'$\lambda_2$ = %0.4f' %
                (zeta * omega + omega * (zeta ** 2 - 1)))
    ax.legend();