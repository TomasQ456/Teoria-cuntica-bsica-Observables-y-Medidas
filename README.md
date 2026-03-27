# Simulador de Mecánica Cuántica — Capítulo 4

Simulación del sistema cuántico descrito en el **Capítulo 4** de  
*Quantum Computing for Computer Scientists* — Yanofsky & Mannucci.

---

## Contenido del Repositorio

```
quantum_chapter4/
├── README.md                  ← este archivo
├── requirements.txt           ← dependencias Python
├── quantum_simulator.py       ← librería principal
└── quantum_chapter4.ipynb    ← notebook Jupyter con todos los ejercicios
```

---

## Librería `quantum_simulator.py`

| Clase | Descripción |
|-------|-------------|
| `QuantumState(amplitudes)` | Vector ket en C^n. Normaliza automáticamente. |
| `Observable(matrix)` | Operador hermitiano. Verifica hermiticidad. |
| `QuantumDynamics(unitaries)` | Secuencia de matrices unitarias. Verifica unitaridad. |
| `QuantumSystem` | Métodos estáticos para sistemas compuestos. |

### Métodos principales

```python
from quantum_simulator import QuantumState, Observable, QuantumDynamics, QuantumSystem
import numpy as np

# ── Estado cuántico ──────────────────────────────────────────
psi = QuantumState([1, 1j, -1, 0])          # normalizado automáticamente
p   = psi.prob_position(0)                  # P(posición 0) = |c_0|²
amp = psi.transition_amplitude(phi)         # ⟨φ|ψ⟩
pt  = psi.transition_probability(phi)       # |⟨φ|ψ⟩|²

# ── Observable ──────────────────────────────────────────────
Sz  = Observable([[1,0],[0,-1]])            # verifica hermiticidad
mu  = Sz.mean(psi)                         # ⟨Sz⟩_ψ
var = Sz.variance(psi)                     # Var(Sz)
eigenvals, probs, eigenstates = Sz.collapse_probabilities(psi)

# ── Dinámica ─────────────────────────────────────────────────
H = (1/np.sqrt(2)) * np.array([[1,1],[1,-1]])
dyn    = QuantumDynamics([H, H, H])         # verifica unitaridad
orbit  = dyn.evolve(psi)                    # lista de estados en t=0,1,2,3
final  = dyn.final_state(psi)

# ── Sistema compuesto ────────────────────────────────────────
composed = QuantumSystem.tensor_product([psi_A, psi_B])
is_sep, factors = QuantumSystem.is_separable(composed, n=2, m=2)
probs_marginal = QuantumSystem.partial_measurement_probs(composed, 2, 2)
```

---

## Navegación del Notebook

El notebook `quantum_chapter4.ipynb` está organizado en **8 secciones**:

| # | Sección | Contenido |
|---|---------|-----------|
| 1 | Configuración | Importaciones e instalación |
| 2 | §4.1 — Estados Cuánticos | **Drill 4.1.1**: probabilidades de posición y transición |
| 3 | §4.2 — Observables | **Drill 4.2.1**: media y varianza de un observable |
| 4 | §4.3 — Medición | **Drill 4.3.1**: autovalores, colapso y distribución |
| 5 | §4.4 — Dinámica | **Drill 4.4.1**: evolución con unitarias |
| 6 | Problemas del Libro | **Ejercicios 4.3.1, 4.3.2, 4.4.1, 4.4.2** |
| 7 | §4.5 — Sistemas Compuestos | Producto tensorial y entrelazamiento |
| 8 | Discusión 4.5.2 y 4.5.3 | Registros de spin, separabilidad y entrelazamiento |

---

## Problemas Resueltos

### Ejercicio 4.3.1
Operador $S_x$ sobre el estado $|\uparrow\rangle$: colapso con probabilidad 50/50
a los autoestados $|\rightarrow\rangle$ y $|\leftarrow\rangle$.

### Ejercicio 4.3.2
Verificación de que $\langle\Omega\rangle_\psi = \sum_i p_i \lambda_i$ coincide con
el cálculo directo $\langle\psi|\Omega|\psi\rangle$.

### Ejercicio 4.4.1
Verificación de que $U_1$, $U_2$ y $U_2 U_1$ son todas matrices unitarias
(producto de sus transpuestas conjugadas = identidad).

### Ejercicio 4.4.2
Evolución de la pelota cuántica con la unitaria 4×4 en 3 pasos de tiempo,
calculando la probabilidad de encontrarla en el punto 3.

---

## Discusión: Ejercicios 4.5.2 y 4.5.3

### 4.5.2 — Registros cuánticos de spin
El espacio de estados de $n$ partículas con spin crece como $\mathbb{C}^{2^n}$.  
Con $n = 50$ qubits se tienen más de $10^{15}$ estados base — imposible para
simulación clásica. Este crecimiento exponencial es la fuente del poder de la
computación cuántica.

### 4.5.3 — Separabilidad de $|\phi\rangle$
El estado $|\phi\rangle = |x_0\rangle\otimes|y_1\rangle + |x_1\rangle\otimes|y_1\rangle$
es **separable** porque la matriz de coeficientes tiene rango 1.
Factoriza como $(|x_0\rangle + |x_1\rangle) \otimes |y_1\rangle$:
la segunda partícula siempre está en $|y_1\rangle$, sin correlaciones cuánticas.

---

*Basado en: Yanofsky, N. S. & Mannucci, M. A. (2008). Quantum Computing for Computer Scientists. Cambridge University Press.*
