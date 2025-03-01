# Physics Simulation: Single Qubit Coupled with a Bath

## Initial State

At t=0, the state of the system is:

$$\rho^{system} (t = 0) = |+\rangle \langle +|$$

The bath's state is thermal:

$$\rho^{bath} (t = 0 ) = \otimes_{i=1}^{N_{bath}} ( w_0|0\rangle_i \langle0|_i + w_1|1\rangle_i \langle1|_i)$$

where $w_0$ and $w_1$ are weights dictated by the Boltzmann distribution:

$$w_0 = \frac{e^{-E_0 / T}}{Z}$$ (weight of $|0\rangle$ bath qubit state)

$$w_1 = \frac{e^{-E_1 / T}}{Z}$$ (weight of $|1\rangle$ bath qubit state)

with the partition function:

$$Z = e^{-E_0 / T} + e^{-E_1 / T}$$

At t=0, the system+bath density matrix is given by:

$$\rho^{system + bath} (t=0) = \rho^{system} (t = 0) \otimes \rho^{bath} (t = 0 )$$

where:


\[
\rho^{system + bath} = \sum\limits_{\substack{(j,k) \in \{0,1\}^{\otimes 2} \\ (i_1, \dots, i_{N_{bath}}) \in \{0,1\}^{\otimes N_{bath}}}} \rho^{system}_{j,k} |j\rangle \langle k| \otimes W_{i_1,i_2, \dots, i_{N_{bath}}} |i_1 \dots i_{N_{bath}}\rangle \langle i_1 \dots i_{N_{bath}}|
\]

---

## Representation

The system+bath state can be represented as a tensor in the space:

$$H_{system} \otimes H_{system}^{\dagger} \otimes H_{1}  \otimes \dots \otimes H_{N_{bath}} \otimes H_{1}^{\dagger} \otimes \dots \otimes H_{N_{bath}}^{\dagger}$$

In this space, the density matrix takes the form:

$$\rho^{system + bath} = \sum\limits_{\substack{{j,k} \in {\{0,1\}^{\otimes 2}} \\ {\{i_1, \dots, i_{N_{bath}}\}}\in {\{0,1\}^{\otimes N_{bath}}}}} \rho^{system}_{j,k}|j\rangle \langle k| \otimes  W_{i_1,i_2, \dots i_{N_{bath}}} |i_1 ... i_{N_{bath}}\rangle \langle i_1 ... i_{N_{bath}}|$$

where $W_{i_1,i_2, \dots i_{N_{bath}}}$ is a product of single-qubit weights:

$$W_{i_1,i_2, \dots i_{N_{bath}}} = \prod \limits_{k=1}^{N_{bath}} w_{i_k} = w_0^{N_{bath} - \sum_k i_k} w_1^{\sum_k i_k}$$

---

## Memory Management

The system+bath state is stored as an `np.ndarray` object `state` of shape `(2,2,N_{bath})`, such that:

\[
state[j,k,i_1,...i_N] = \rho^{system}_{j,k} W_{i_1,i_2, \dots i_{N_{bath}}}
\]

Introducing the super-index $I = (i_1,i_2, \dots i_{N_{bath}})$ for simplicity:

\[
\text{state}[j,k,I] = \rho^{\text{system}}_{j,k} W_{I}
\]

*Note: In the program, $I$ is 0-indexed, while in the formulas above, $I$ is 1-indexed.*

---

## Hamiltonian

The Hamiltonian of the system is given by:

$$H = \sum\limits_{q=1}^{N_{bath}} \gamma_q Z_{system} Z_{q}$$

where $q$ is the bath index.

The evolution operator is expressed as:

$$ U(t) = \exp(-iHt) $$

Since:

$$Z_{system} Z_{q} |j\rangle \otimes |i_1,...i_{N_{bath}}\rangle = (-1)^{(j + i_q)} |j\rangle \otimes |i_1,...i_{N_{bath}}\rangle$$

We arrive at:

$$ U(t) |j\rangle \langle k| \otimes |I\rangle \langle I| U(t)^{\dagger} = e^{-it\{(-1)^{j} - (-1)^{k}\} \sum_q \gamma_q (-1)^q} |j\rangle \langle k| \otimes |I\rangle \langle I| $$
