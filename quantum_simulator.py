"""
quantum_simulator.py
====================
Simulador de sistemas cuánticos basado en el Capítulo 4 de
"Quantum Computing for Computer Scientists" — Yanofsky & Mannucci.

Módulos
-------
- QuantumState    : estado ket de una partícula en posiciones discretas
- Observable      : operador hermitiano con cálculo de media, varianza y autovalores
- QuantumDynamics : evolución temporal mediante matrices unitarias
- QuantumSystem   : sistema compuesto (producto tensorial) y entrelazamiento
"""

import numpy as np
from numpy import linalg as la
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _is_hermitian(M: np.ndarray, tol: float = 1e-9) -> bool:
    """Verifica si la matriz M es hermitiana (M == M†)."""
    return np.allclose(M, M.conj().T, atol=tol)


def _is_unitary(M: np.ndarray, tol: float = 1e-9) -> bool:
    """Verifica si la matriz M es unitaria (M·M† == I)."""
    n = M.shape[0]
    return np.allclose(M @ M.conj().T, np.eye(n), atol=tol)


def _normalize(v: np.ndarray) -> np.ndarray:
    """Devuelve el vector v normalizado."""
    norm = la.norm(v)
    if norm < 1e-12:
        raise ValueError("El vector cero no puede normalizarse.")
    return v / norm


# ──────────────────────────────────────────────
# 4.1  Estado cuántico (partícula en una línea)
# ──────────────────────────────────────────────

class QuantumState:
    """
    Representa un vector ket |ψ⟩ en C^n.

    Parámetros
    ----------
    amplitudes : array-like de complejos de longitud n.
    normalize  : si True, normaliza automáticamente las amplitudes.
    """

    def __init__(self, amplitudes, normalize: bool = True):
        arr = np.array(amplitudes, dtype=complex)
        if arr.ndim != 1:
            raise ValueError("Las amplitudes deben ser un vector 1D.")
        if normalize:
            arr = _normalize(arr)
        else:
            # Solo verificamos que ya esté normalizado
            if not np.isclose(la.norm(arr), 1.0):
                raise ValueError(
                    "El vector no está normalizado. "
                    "Usa normalize=True para normalizarlo automáticamente."
                )
        self._amp = arr

    # ── propiedades básicas ───────────────────

    @property
    def n(self) -> int:
        """Número de posiciones del sistema."""
        return len(self._amp)

    @property
    def amplitudes(self) -> np.ndarray:
        return self._amp.copy()

    def bra(self) -> np.ndarray:
        """Devuelve ⟨ψ| = (|ψ⟩)†  (fila de conjugados)."""
        return self._amp.conj()

    # ── Drill 4.1.1 — probabilidad en una posición ──

    def prob_position(self, i: int) -> float:
        """
        Probabilidad de encontrar la partícula en la posición i.
        P(i) = |c_i|²
        """
        if not (0 <= i < self.n):
            raise IndexError(f"Índice {i} fuera de rango [0, {self.n-1}].")
        return float(abs(self._amp[i]) ** 2)

    def all_probabilities(self) -> np.ndarray:
        """Vector de probabilidades para todas las posiciones."""
        return np.abs(self._amp) ** 2

    # ── Drill 4.1.1 — amplitud / probabilidad de transición ──

    def transition_amplitude(self, other: "QuantumState") -> complex:
        """
        Amplitud de transición ⟨φ|ψ⟩ desde |ψ⟩=self hacia |φ⟩=other.
        Ecuación (4.31) del libro.
        """
        if self.n != other.n:
            raise ValueError("Los estados deben tener la misma dimensión.")
        return complex(other.bra() @ self._amp)

    def transition_probability(self, other: "QuantumState") -> float:
        """
        Probabilidad de transitar de |ψ⟩=self a |φ⟩=other tras una observación.
        P = |⟨φ|ψ⟩|²
        """
        return abs(self.transition_amplitude(other)) ** 2

    # ── utilidades ────────────────────────────

    def __repr__(self) -> str:
        formatted = ", ".join(
            f"{a.real:.4f}{a.imag:+.4f}j" if a.imag != 0 else f"{a.real:.4f}"
            for a in self._amp
        )
        return f"|ψ⟩ = [{formatted}]ᵀ"

    @classmethod
    def basis(cls, n: int, i: int) -> "QuantumState":
        """Devuelve el i-ésimo vector de la base canónica de C^n."""
        v = np.zeros(n, dtype=complex)
        v[i] = 1.0
        return cls(v, normalize=False)


# ──────────────────────────────────────────────
# 4.2 / 4.3  Observable (operador hermitiano)
# ──────────────────────────────────────────────

class Observable:
    """
    Observable cuántico representado por una matriz hermitiana Ω.

    Parámetros
    ----------
    matrix : array 2D cuadrado complejo.
    """

    def __init__(self, matrix):
        M = np.array(matrix, dtype=complex)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("El observable debe ser una matriz cuadrada.")
        if not _is_hermitian(M):
            raise ValueError(
                "La matriz NO es hermitiana. Los observables deben ser hermitianos."
            )
        self._M = M

    @property
    def matrix(self) -> np.ndarray:
        return self._M.copy()

    @property
    def n(self) -> int:
        return self._M.shape[0]

    # ── Drill 4.2.1 — media y varianza ───────

    def mean(self, state: QuantumState) -> float:
        """
        Valor esperado (media) del observable en el estado |ψ⟩.
        ⟨Ω⟩_ψ = ⟨ψ|Ω|ψ⟩  (siempre real para hermitianos)
        """
        if state.n != self.n:
            raise ValueError("Dimensiones incompatibles.")
        val = state.bra() @ self._M @ state.amplitudes
        return float(val.real)

    def variance(self, state: QuantumState) -> float:
        """
        Varianza del observable en el estado |ψ⟩.
        Var(Ω) = ⟨Ω²⟩_ψ − ⟨Ω⟩_ψ²
        """
        mu = self.mean(state)
        # ⟨Ω²⟩
        psi = state.amplitudes
        mean_sq = float((state.bra() @ self._M @ self._M @ psi).real)
        return mean_sq - mu ** 2

    # ── Drill 4.3.1 — autovalores y probabilidades de colapso ──

    def eigendecomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devuelve (eigenvalues, eigenvectors_normalized).
        Los vectores propios son columnas de la matriz retornada.
        """
        vals, vecs = la.eigh(self._M)   # eigh garantiza reales para hermitianas
        # normalizar columnas (eigh ya las devuelve normalizadas, pero reafirmamos)
        vecs = vecs / la.norm(vecs, axis=0)
        return vals.real, vecs

    def collapse_probabilities(
        self, state: QuantumState
    ) -> Tuple[np.ndarray, np.ndarray, List[QuantumState]]:
        """
        Calcula probabilidad de colapso a cada autovector.

        Retorna
        -------
        eigenvalues : array de autovalores reales
        probs       : probabilidad de colapsar a cada autovector
        eigenstates : lista de QuantumState correspondientes
        """
        if state.n != self.n:
            raise ValueError("Dimensiones incompatibles.")
        vals, vecs = self.eigendecomposition()
        probs = []
        eigenstates = []
        for j in range(self.n):
            ev = QuantumState(vecs[:, j], normalize=False)
            p = state.transition_probability(ev)
            probs.append(p)
            eigenstates.append(ev)
        return vals, np.array(probs), eigenstates

    def __repr__(self) -> str:
        return f"Observable(n={self.n}, hermitian=True)\n{self._M}"


# ──────────────────────────────────────────────
# 4.4  Dinámica cuántica (operadores unitarios)
# ──────────────────────────────────────────────

class QuantumDynamics:
    """
    Evolución temporal de un sistema cuántico mediante operadores unitarios.

    Dado un estado inicial |ψ₀⟩ y una secuencia U[t₀], U[t₁], …, U[t_{n-1}],
    calcula el estado final.

    |ψ(n)⟩ = U[t_{n-1}] · … · U[t₁] · U[t₀] · |ψ₀⟩
    """

    def __init__(self, unitary_matrices: List):
        matrices = [np.array(U, dtype=complex) for U in unitary_matrices]
        for k, U in enumerate(matrices):
            if not _is_unitary(U):
                raise ValueError(
                    f"La matriz en el paso {k} NO es unitaria."
                )
        self._Us = matrices

    def evolve(self, initial_state: QuantumState) -> List[QuantumState]:
        """
        Aplica la secuencia de unitarias al estado inicial.

        Retorna la lista de estados en cada instante (incluido el inicial).
        """
        states = [initial_state]
        current = initial_state.amplitudes
        for U in self._Us:
            if U.shape[1] != len(current):
                raise ValueError(
                    f"Dimensión de la unitaria ({U.shape}) incompatible con "
                    f"el estado ({len(current)})."
                )
            current = U @ current
            states.append(QuantumState(current, normalize=False))
        return states

    def final_state(self, initial_state: QuantumState) -> QuantumState:
        """Devuelve directamente el estado final."""
        return self.evolve(initial_state)[-1]

    def __repr__(self) -> str:
        return f"QuantumDynamics(steps={len(self._Us)})"


# ──────────────────────────────────────────────
# 4.5  Sistema compuesto — producto tensorial
# ──────────────────────────────────────────────

class QuantumSystem:
    """
    Sistema cuántico compuesto de múltiples partículas.
    El espacio de estados es el producto tensorial de los espacios individuales.
    """

    @staticmethod
    def tensor_product(
        states: List[QuantumState], normalize: bool = True
    ) -> QuantumState:
        """
        Calcula |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ … ⊗ |ψₖ⟩.
        """
        result = states[0].amplitudes
        for s in states[1:]:
            result = np.kron(result, s.amplitudes)
        return QuantumState(result, normalize=normalize)

    @staticmethod
    def is_separable(
        state: QuantumState, n: int, m: int, tol: float = 1e-9
    ) -> Tuple[bool, Optional[Tuple[QuantumState, QuantumState]]]:
        """
        Comprueba si el estado de un sistema bipartito (n × m) es separable.

        Usa la descomposición en valores singulares de la matriz de coeficientes.
        El estado es separable ↔ la matriz de coeficientes tiene rango 1.

        Parámetros
        ----------
        state : estado del sistema compuesto de dimensión n*m
        n, m  : dimensiones de los subsistemas

        Retorna
        -------
        (True, (state_A, state_B)) si es separable, (False, None) si entrelazado.
        """
        if state.n != n * m:
            raise ValueError(
                f"El estado debe tener dimensión {n}*{m} = {n*m}. "
                f"Tiene {state.n}."
            )
        # Reescribimos los coeficientes como matriz n×m
        C = state.amplitudes.reshape(n, m)
        # SVD
        U_svd, S, Vh = la.svd(C)
        # Separable ↔ solo un valor singular no nulo
        rank = np.sum(S > tol)
        if rank == 1:
            # state_A ∝ col 0 de U_svd, state_B ∝ fila 0 de Vh
            a = QuantumState(U_svd[:, 0], normalize=True)
            b = QuantumState(Vh[0, :].conj(), normalize=True)
            return True, (a, b)
        return False, None

    @staticmethod
    def partial_measurement_probs(
        state: QuantumState, n: int, m: int, measure_first: bool = True
    ) -> np.ndarray:
        """
        Calcula las probabilidades marginales de medir el primer (o segundo)
        subsistema en el estado base canónico.

        Parámetros
        ----------
        n, m         : dimensiones de subsistemas A (n) y B (m)
        measure_first: si True, mide el subsistema A; si False, mide B
        """
        C = state.amplitudes.reshape(n, m)
        if measure_first:
            # P(i) = Σ_j |C[i,j]|²
            return np.sum(np.abs(C) ** 2, axis=1)
        else:
            # P(j) = Σ_i |C[i,j]|²
            return np.sum(np.abs(C) ** 2, axis=0)
