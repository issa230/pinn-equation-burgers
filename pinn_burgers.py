# %% [markdown]
# # PINN : résoudre l'équation de Burgers avec un réseau de neurones
#
# On cherche u(x, t) tel que :
#
#     u_t + u * u_x - nu * u_xx = 0        pour x dans [-1, 1] et t dans [0, 1]
#     u(x, 0)  = -sin(pi * x)              (condition initiale)
#     u(-1, t) = u(1, t) = 0               (conditions aux limites)
#     nu = 0.01 / pi                       (viscosité, très petite -> apparition d'un choc)
#
# Idée du PINN : un réseau de neurones prend (x, t) en entrée et renvoie u(x, t).
# On l'entraîne pour qu'il respecte en même temps l'EDP, la condition initiale
# et les conditions aux limites. Aucune donnée « solution » n'est utilisée :
# c'est la physique (l'équation) qui sert de professeur.

# %%
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Reproductibilité : mêmes points tirés au hasard et mêmes poids initiaux à chaque exécution
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Calcul sur : {device}")
os.makedirs("figures", exist_ok=True)

# %% [markdown]
# ## 1. Le problème physique

# %%
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0
NU = 0.01 / np.pi

N_F = 10_000   # points de collocation (là où on impose l'EDP)
N_0 = 200      # points sur la condition initiale (t = 0)
N_B = 200      # points sur chaque bord (x = -1 et x = 1)

# Entraînement
N_ADAM = 2_000      # itérations d'Adam (phase d'exploration)
LR_ADAM = 1e-3
N_LBFGS = 5_000     # évaluations max de L-BFGS (phase de précision)

# %% [markdown]
# ## 2. Les points d'entraînement
#
# - **Collocation** : points tirés au hasard dans tout le domaine (x, t). On y impose l'EDP.
# - **Condition initiale** : points sur la ligne t = 0. On y impose u = -sin(pi x).
# - **Conditions aux limites** : points sur x = -1 et x = 1. On y impose u = 0.


# %%
def to_tensor(array, requires_grad=False):
    """Convertit un tableau numpy en tenseur PyTorch (float32) sur le bon device."""
    return torch.tensor(array, dtype=torch.float32, device=device, requires_grad=requires_grad)


# Points de collocation (EDP) : tirage uniforme dans [-1, 1] x [0, 1]
x_f_np = np.random.uniform(X_MIN, X_MAX, size=(N_F, 1))
t_f_np = np.random.uniform(T_MIN, T_MAX, size=(N_F, 1))

# Condition initiale : t = 0
x_0_np = np.linspace(X_MIN, X_MAX, N_0)[:, None]
t_0_np = np.zeros_like(x_0_np)
u_0_np = -np.sin(np.pi * x_0_np)

# Conditions aux limites : x = -1 et x = 1, pour tous les temps
t_b_np = np.linspace(T_MIN, T_MAX, N_B)[:, None]
x_b_np = np.vstack([np.full_like(t_b_np, X_MIN), np.full_like(t_b_np, X_MAX)])
t_b_np = np.vstack([t_b_np, t_b_np])
u_b_np = np.zeros_like(x_b_np)

# Tenseurs PyTorch. requires_grad=True sur les points de collocation :
# on aura besoin de dériver u par rapport à x et t en ces points.
x_f = to_tensor(x_f_np, requires_grad=True)
t_f = to_tensor(t_f_np, requires_grad=True)
x_0, t_0, u_0 = to_tensor(x_0_np), to_tensor(t_0_np), to_tensor(u_0_np)
x_b, t_b, u_b = to_tensor(x_b_np), to_tensor(t_b_np), to_tensor(u_b_np)

# %% [markdown]
# ## 3. Le réseau de neurones
#
# Entrée (x, t) -> 4 couches cachées de 50 neurones (tanh) -> sortie u(x, t).
#
# Pourquoi tanh ? Parce qu'elle est lisse (dérivable autant de fois qu'on veut).
# Avec ReLU, la dérivée seconde vaut 0 presque partout : le terme nu * u_xx disparaîtrait.


# %%
class PINN(nn.Module):
    def __init__(self, couches):
        super().__init__()
        self.activation = nn.Tanh()
        self.couches = nn.ModuleList(
            [nn.Linear(couches[i], couches[i + 1]) for i in range(len(couches) - 1)]
        )
        # Initialisation de Xavier : adaptée à tanh, évite que les signaux explosent ou s'éteignent
        for couche in self.couches:
            nn.init.xavier_normal_(couche.weight)
            nn.init.zeros_(couche.bias)

    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)          # (N, 2)
        for couche in self.couches[:-1]:
            z = self.activation(couche(z))
        return self.couches[-1](z)            # (N, 1) : pas d'activation en sortie


model = PINN([2, 50, 50, 50, 50, 1]).to(device)
print(model)
print(f"Nombre de paramètres : {sum(p.numel() for p in model.parameters())}")

# %% [markdown]
# ## 4. La fonction de perte (loss) : la physique dans le code
#
# - Résidu de l'EDP : f = u_t + u * u_x - nu * u_xx (doit valoir 0)
# - Erreur sur la condition initiale : u(x, 0) - (-sin(pi x))
# - Erreur sur les bords : u(+-1, t) - 0
#
# Les dérivées u_t, u_x, u_xx sont calculées par **différentiation automatique**
# (torch.autograd.grad) : ce sont les dérivées exactes du réseau, pas une approximation
# par différences finies.


# %%
def derivee(sortie, entree):
    """Dérivée de `sortie` par rapport à `entree` (create_graph=True pour pouvoir re-dériver)."""
    return torch.autograd.grad(
        sortie, entree, grad_outputs=torch.ones_like(sortie), create_graph=True
    )[0]


def residu_edp(model, x, t):
    u = model(x, t)
    u_t = derivee(u, t)
    u_x = derivee(u, x)
    u_xx = derivee(u_x, x)
    return u_t + u * u_x - NU * u_xx


def calcul_loss(model):
    loss_edp = torch.mean(residu_edp(model, x_f, t_f) ** 2)
    loss_ci = torch.mean((model(x_0, t_0) - u_0) ** 2)
    loss_cl = torch.mean((model(x_b, t_b) - u_b) ** 2)
    loss = loss_edp + loss_ci + loss_cl
    return loss, (loss_edp.item(), loss_ci.item(), loss_cl.item())


# %% [markdown]
# ## 5. Entraînement en 2 temps : Adam puis L-BFGS
#
# 1. **Adam** (gradient du 1er ordre, pas adaptatif) : robuste au départ, quand les poids
#    sont aléatoires. Il fait vite baisser la loss mais devient lent près du minimum.
# 2. **L-BFGS** (quasi-Newton) : utilise l'historique des gradients pour estimer la courbure
#    de la loss. Très précis pour « finir le travail » quand on est déjà dans la bonne vallée.

# %%
historique = []   # valeur de la loss à chaque étape (Adam puis L-BFGS)

# ---- Phase 1 : Adam ----
t_debut = time.time()
optimizer_adam = torch.optim.Adam(model.parameters(), lr=LR_ADAM)

for it in range(1, N_ADAM + 1):
    optimizer_adam.zero_grad()
    loss, (l_edp, l_ci, l_cl) = calcul_loss(model)
    loss.backward()
    optimizer_adam.step()
    historique.append(loss.item())
    if it % 500 == 0 or it == 1:
        print(f"[Adam]   it {it:5d} | loss = {loss.item():.3e} "
              f"(EDP {l_edp:.1e}, CI {l_ci:.1e}, CL {l_cl:.1e})")

fin_adam = len(historique)

# ---- Phase 2 : L-BFGS ----
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,                         # pas de base ; la recherche linéaire l'ajuste
    max_iter=N_LBFGS,               # nombre max d'itérations
    max_eval=N_LBFGS,               # nombre max d'évaluations de la loss
    history_size=50,                # nombre de gradients passés gardés en mémoire
    tolerance_grad=1e-5,            # arrêt si le gradient devient minuscule
    tolerance_change=1e-9,          # arrêt si la loss ne bouge plus
    line_search_fn="strong_wolfe",  # recherche linéaire : indispensable pour la stabilité
)


def closure():
    """L-BFGS ré-évalue la loss plusieurs fois par itération : on lui donne une fonction."""
    optimizer_lbfgs.zero_grad()
    loss, _ = calcul_loss(model)
    loss.backward()
    historique.append(loss.item())
    n = len(historique) - fin_adam
    if n % 500 == 0:
        print(f"[L-BFGS] éval {n:5d} | loss = {loss.item():.3e}")
    return loss


optimizer_lbfgs.step(closure)

loss_finale, (l_edp, l_ci, l_cl) = calcul_loss(model)
print(f"\nLoss finale : {loss_finale.item():.3e} "
      f"(EDP {l_edp:.1e}, CI {l_ci:.1e}, CL {l_cl:.1e})")
print(f"Temps d'entraînement : {time.time() - t_debut:.0f} s")

# %% [markdown]
# ## 6. Solution exacte (transformation de Cole-Hopf) pour vérifier le PINN
#
# Burgers a une solution exacte sous forme d'intégrale :
#
#     u(x, t) = ∫ ((x - y) / t) exp(-G / (2 nu)) dy  /  ∫ exp(-G / (2 nu)) dy
#     avec G(y) = (cos(pi y) - 1) / pi + (x - y)^2 / (2 t)
#
# On calcule ces intégrales numériquement (méthode des trapèzes sur une grille fine).

# %%
_trapeze = getattr(np, "trapezoid", None) or np.trapz   # compatible numpy 1.x et 2.x


def solution_exacte(x, t, nu=NU, n_y=4000):
    """Solution exacte de Burgers (u0 = -sin(pi x)) en tous les points x, au temps t."""
    x = np.asarray(x, dtype=float)
    if t == 0:
        return -np.sin(np.pi * x)
    n_y = max(n_y, int(24.0 / np.sqrt(2.0 * nu * t)))     # grille assez fine si t est petit
    y = np.linspace(-3.0, 3.0, n_y)                       # variable d'intégration
    G = (np.cos(np.pi * y) - 1.0) / np.pi + (x[:, None] - y) ** 2 / (2.0 * t)
    expo = -G / (2.0 * nu)
    expo -= expo.max(axis=1, keepdims=True)               # astuce : évite l'overflow de exp
    poids = np.exp(expo)
    num = _trapeze((x[:, None] - y) / t * poids, y, axis=1)
    den = _trapeze(poids, y, axis=1)
    return num / den


# %% [markdown]
# ## 7. Résultats

# %%
N_X, N_T = 256, 100
x_grid = np.linspace(X_MIN, X_MAX, N_X)
t_grid = np.linspace(T_MIN, T_MAX, N_T)
X, T = np.meshgrid(x_grid, t_grid)

model.eval()
with torch.no_grad():
    u_pred = model(to_tensor(X.reshape(-1, 1)), to_tensor(T.reshape(-1, 1)))
    u_pred = u_pred.cpu().numpy().reshape(N_T, N_X)

u_exact = np.array([solution_exacte(x_grid, tt) for tt in t_grid])
erreur_l2 = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
print(f"Erreur relative L2 (PINN vs solution exacte) : {erreur_l2:.3e}")

# --- Figure 1 : carte de la solution prédite ---
fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.pcolormesh(T, X, u_pred, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
fig.colorbar(im, ax=ax, label="u(x, t)")
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_title("Solution prédite par le PINN")
fig.savefig("figures/solution_pinn.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 2 : comparaison avec la solution exacte à 3 instants ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, t_val in zip(axes, [0.25, 0.50, 0.75]):
    k = np.argmin(np.abs(t_grid - t_val))
    ax.plot(x_grid, u_exact[k], color="#2a78d6", lw=3, label="Solution exacte")
    ax.plot(x_grid, u_pred[k], "--", color="#eb6834", lw=2.5, label="PINN")
    ax.set_title(f"t = {t_grid[k]:.2f}")
    ax.set_xlabel("x")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("u(x, t)")
axes[0].legend()
fig.suptitle(f"PINN vs solution exacte (erreur relative L2 = {erreur_l2:.1e})")
fig.savefig("figures/comparaison_exacte.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 3 : évolution de la loss ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(range(1, fin_adam + 1), historique[:fin_adam], color="#2a78d6", label="Adam")
ax.semilogy(range(fin_adam + 1, len(historique) + 1), historique[fin_adam:],
            color="#eb6834", label="L-BFGS")
ax.set_xlabel("Itérations (Adam) puis évaluations (L-BFGS)")
ax.set_ylabel("Loss (échelle log)")
ax.set_title("Évolution de la loss pendant l'entraînement")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.savefig("figures/loss.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Animation : formation du choc

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
ligne_exacte, = ax.plot([], [], color="#2a78d6", lw=3, label="Solution exacte")
ligne_pinn, = ax.plot([], [], "--", color="#eb6834", lw=2.5, label="PINN")
texte = ax.text(0.03, 0.92, "", transform=ax.transAxes, fontsize=12)
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")


def maj(k):
    ligne_exacte.set_data(x_grid, u_exact[k])
    ligne_pinn.set_data(x_grid, u_pred[k])
    texte.set_text(f"t = {t_grid[k]:.2f}")
    return ligne_exacte, ligne_pinn, texte


anim = FuncAnimation(fig, maj, frames=N_T, interval=80, blit=True)
anim.save("figures/burgers_animation.gif", writer=PillowWriter(fps=12))
plt.close(fig)
print("Figures enregistrées dans le dossier figures/")
