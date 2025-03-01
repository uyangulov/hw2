import numpy as np
import matplotlib.pyplot as plt

def init_system_state():
    """
    Initialize the system state as a pure state |+> = (|0> + |1>) / sqrt(2)
    and compute its density matrix representation.
    
    Returns:
        np.ndarray: The density matrix of the system.
    """
    sys_psi = np.array([1, 1]/np.sqrt(2), dtype=np.complex128)
    sys_rho = np.outer(sys_psi, np.conj(sys_psi))
    return sys_rho

def I_bitwise(N_bath):
    """
    Compute the bitwise representation of integers from 0 to 2^N_bath - 1.
    
    Args:
        N_bath (int): Number of bath qubits.
    
    Returns:
        np.ndarray: Bitwise representation of integers as a matrix.
    """
    I = np.arange(2**N_bath)
    bit_matrix = (I[:, None] >> np.arange(N_bath-1, -1, -1)) & 1
    return bit_matrix

def init_env_state(N_bath, T, E0, E1):
    """
    Initialize the environment (bath) state with a thermal distribution.
    
    Args:
        N_bath (int): Number of bath qubits.
        T (float): Temperature of the bath.
        E0 (float): Energy level 0.
        E1 (float): Energy level 1.
    
    Returns:
        np.ndarray: Probability distribution over bath states.
    """
    w0, w1 = np.exp(-E0/T), np.exp(-E1/T)
    Z = w0 + w1
    w0 = w0 / Z
    w1 = w1 / Z

    I = I_bitwise(N_bath)
    sum_indices = I.sum(axis=1)
    W_I = (w0 ** sum_indices) * (w1 ** (N_bath - sum_indices))

    return W_I

def init_system_env_state(N_bath, T, E0, E1):
    """
    Initialize the combined system-environment state.
    
    Args:
        N_bath (int): Number of bath qubits.
        T (float): Temperature of the bath.
        E0 (float): Energy level 0.
        E1 (float): Energy level 1.
    
    Returns:
        np.ndarray: Combined state tensor.
    """
    sys = init_system_state()
    env = init_env_state(N_bath, T, E0, E1)
    state = sys[..., None] * env[None, None, :]
    return state

def hamiltonian(gammas):
    """
    Construct the Hamiltonian describing system-environment interactions.
    
    Args:
        gammas (np.ndarray): Coupling strengths for each bath qubit.
    
    Returns:
        np.ndarray: Hamiltonian interaction terms.
    """
    N_bath = gammas.shape[0]
    jj, kk = np.meshgrid([0, 1], [0, 1])
    sys_phase_mask = (-1)**jj - (-1)**kk
    I = I_bitwise(N_bath)
    bath_phase_mask = (gammas * (-1)**I).sum(axis=1)
    phase_mask = sys_phase_mask[..., None] * bath_phase_mask[None, None, :]
    return phase_mask

def evolve(state, H, dt):
    """
    Evolve the state according to the Hamiltonian.
    
    Args:
        state (np.ndarray): Current state.
        H (np.ndarray): Hamiltonian of the system+bath.
        dt (float): Time step.
    
    Returns:
        np.ndarray: Updated state.
    """
    return state * np.exp(-1j * H * dt)

def apply_pi_pulse(state):
    """
    Apply a pi-pulse (bit-flip operation) to the system.
    
    Args:
        state (np.ndarray): Current system+bath state.
    
    Returns:
        np.ndarray: State with bit-flip applied to system.
    """
    state_m = state[::-1, ::-1, :]
    return state_m

def system_rho(state):
    """
    Compute density matrix of system by taking trace over bath indeces.
    
    Args:
        state (np.ndarray): System-bath state.
    
    Returns:
        np.ndarray: Density matrix of the system.
    """
    return np.sum(state, axis=tuple(range(2, state.ndim)))

def purity(system_rho):
    """
    Compute the purity of the system state.
    
    Args:
        system_rho (np.ndarray): Density matrix of the system.
    
    Returns:
        float: Purity of the system state.
    """
    return np.trace(system_rho @ system_rho)

def fidelity(system_rho1, system_rho2):
    """
    Compute the fidelity between two system states.
    
    Args:
        system_rho1 (np.ndarray): First density matrix.
        system_rho2 (np.ndarray): Second density matrix.
    
    Returns:
        float: Fidelity between the two states.
    """
    return np.trace(system_rho1 @ np.transpose(system_rho2.conj()))



# Simulation parameters
tau = .5
N_t = 1000
timesteps = np.linspace(0, 2*tau, N_t)
dt = np.diff(timesteps)[0]
N_bath = 10
T = 5
E0 = .1
E1 = .9


fidelities = np.zeros_like(timesteps)
purities = np.zeros_like(timesteps)
fidelities_no_pulse = np.zeros_like(timesteps)
purities_no_pulse = np.zeros_like(timesteps)

state = init_system_env_state(N_bath, T, E0, E1)
state_no_pulse = np.copy(state)
init_sys_rho = init_system_state()
gammas = np.random.rand(N_bath)
H = hamiltonian(gammas)

for i, timestep in enumerate(timesteps):
    sys_rho = system_rho(state)
    sys_rho_no_pulse = system_rho(state_no_pulse)
    
    purities[i] = purity(sys_rho)
    fidelities[i] = fidelity(sys_rho, init_sys_rho)
    purities_no_pulse[i] = purity(sys_rho_no_pulse)
    fidelities_no_pulse[i] = fidelity(sys_rho_no_pulse, init_sys_rho)
    
    if (i == timesteps.shape[0] // 2):
        state = apply_pi_pulse(state)
    
    state = evolve(state, H, dt)
    state_no_pulse = evolve(state_no_pulse, H, dt)

plt.plot(timesteps / tau, purities, label='purity (pi-pulse)', color = 'r')
plt.plot(timesteps / tau, fidelities, label='fidelity (pi-pulse)', color = 'b')
plt.plot(timesteps / tau, purities_no_pulse, label='purity (no pi-pulse)', linestyle='dashed', color = 'r')
plt.plot(timesteps / tau, fidelities_no_pulse, label='fidelity (no pi-pulse)', linestyle='dashed', color = 'b')
plt.xlabel(r"t/$\tau$")
plt.grid(ls=':')
plt.legend()
plt.show()
