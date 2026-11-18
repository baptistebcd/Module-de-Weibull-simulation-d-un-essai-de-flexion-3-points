"""
Simulation de Weibull : essai de flexion 3 points d'une éprouvette céramique
============================================================================

Lancer :   streamlit run simulation_weibull.py

Principe
--------
1. L'éprouvette est découpée en petites mailles (dx x dy x b).
2. Chaque maille reçoit une résistance locale tirée au hasard dans une loi de Weibull
   (module m). La loi est ajustée au volume de la maille (modèle du maillon le plus faible).
3. En flexion 3 points la contrainte de traction dans chaque maille est proportionnelle
   à la force F : sigma_i = s_i * F  (s_i en MPa/N, théorie des poutres).
4. L'éprouvette casse quand la première maille atteint sa résistance locale :
        F_rupture = min_i ( sigma_local_i / s_i )
5. On répète le tirage N fois (Monte-Carlo) pour obtenir la distribution des forces à rupture.

Calibrage : sigma_th est la contrainte de rupture CARACTÉRISTIQUE de l'éprouvette entière
(63,2 % de ruptures). F_ref = 2 b h^2 sigma_th / (3 L) est la force RDM correspondante.

Hypothèses et limites : voir l'encadré "Hypothèses du modèle" dans l'application.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

V0 = 1.0  # volume de référence du modèle de Weibull (mm^3)


# --------------------------------------------------------------------------------------
# Moteur de calcul
# --------------------------------------------------------------------------------------
def grille(Lt, b, h, L, dx, ny):
    """Maillage de l'éprouvette et contrainte par newton (MPa/N) en chaque maille.

    x : position le long de l'éprouvette (0 à Lt) ; y : profondeur (0 = face chargée,
    h = face inférieure tendue). Charge au milieu (xc = Lt/2), appuis en xc +/- L/2.
    """
    nx = max(1, int(round(Lt / dx)))
    dx = Lt / nx
    dy = h / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)  # formes (ny, nx)
    xc = Lt / 2
    I = b * h**3 / 12
    bras = np.clip(L / 2 - np.abs(X - xc), 0, None)  # 2M/F entre les appuis
    s_signe = 0.5 * bras * (Y - h / 2) / I  # >0 : traction
    masque = s_signe > 0
    return dict(Lt=Lt, b=b, h=h, L=L, xc=xc, dx=dx, dy=dy, nx=nx, ny=ny, X=X, Y=Y,
                s_signe=s_signe, masque=masque, s_t=s_signe[masque], V=dx * dy * b)


def calibrer(g, m, F_ref):
    """sigma_L tel que la force caractéristique de l'éprouvette vaille F_ref."""
    return F_ref * (np.sum(g["V"] / V0 * g["s_t"] ** m)) ** (1.0 / m)


def echelle_locale(g, m, sigma_L):
    """Paramètre d'échelle de Weibull d'une maille (ajusté à son volume)."""
    return sigma_L * (V0 / g["V"]) ** (1.0 / m)


def simuler(g, m, sigma_L, n, graine, paquet=500):
    """Monte-Carlo : n éprouvettes. Retourne forces à rupture et indices des mailles critiques."""
    rng = np.random.default_rng(graine)
    s_t = g["s_t"]
    ech = echelle_locale(g, m, sigma_L)
    F = np.empty(n)
    idx = np.empty(n, dtype=int)
    for a in range(0, n, paquet):
        k = min(paquet, n - a)
        U = rng.random((k, s_t.size))
        r = ech * (-np.log1p(-U)) ** (1.0 / m) / s_t  # force qui casserait chaque maille
        j = r.argmin(axis=1)
        F[a:a + k] = r[np.arange(k), j]
        idx[a:a + k] = j
    return F, idx


def eprouvette_unique(g, m, sigma_L, graine):
    """Une éprouvette : champ de résistances locales, force à rupture, maille critique."""
    rng = np.random.default_rng(graine)
    U = rng.random((g["ny"], g["nx"]))
    sig_loc = echelle_locale(g, m, sigma_L) * (-np.log1p(-U)) ** (1.0 / m)
    r = np.full(U.shape, np.inf)
    r[g["masque"]] = sig_loc[g["masque"]] / g["s_signe"][g["masque"]]
    j = np.unravel_index(np.argmin(r), r.shape)
    return dict(U=U, sig_loc=sig_loc, F=r[j], j=j, xc_crit=g["X"][j], yc_crit=g["Y"][j])


def ajuster_regression(F):
    """Droite de Weibull par régression, probabilité (i - 0,5)/N. Retourne m, F63."""
    F = np.sort(np.asarray(F))
    P = (np.arange(1, F.size + 1) - 0.5) / F.size
    x, y = np.log(F), np.log(-np.log1p(-P))
    m, c = np.polyfit(x, y, 1)
    return m, float(np.exp(-c / m))


def regression_lignes(Fs):
    """Module de Weibull par régression pour chaque ligne d'un tableau (K, n)."""
    Fs = np.sort(Fs, axis=1)
    n = Fs.shape[1]
    P = (np.arange(1, n + 1) - 0.5) / n
    y = np.log(-np.log1p(-P))
    x = np.log(Fs)
    xm = x.mean(axis=1, keepdims=True)
    return ((x - xm) * (y - y.mean())).sum(axis=1) / ((x - xm) ** 2).sum(axis=1)


def mle_lignes(Fs):
    """Module de Weibull par maximum de vraisemblance (bissection) pour chaque ligne."""
    x = Fs / Fs.max(axis=1, keepdims=True)
    lx = np.log(x)
    lo = np.full(len(x), 0.2)
    hi = np.full(len(x), 300.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        w = x ** mid[:, None]
        f = 1 / mid + lx.mean(axis=1) - (w * lx).sum(axis=1) / w.sum(axis=1)
        lo = np.where(f > 0, mid, lo)
        hi = np.where(f > 0, hi, mid)
    return (lo + hi) / 2


def mle(F):
    """Maximum de vraisemblance sur un échantillon : retourne m et F63."""
    F = np.asarray(F)[None, :]
    m = float(mle_lignes(F)[0])
    return m, float(np.mean(F ** m) ** (1 / m))


def pdf_weibull(t, m, c):
    return (m / c) * (t / c) ** (m - 1) * np.exp(-((t / c) ** m))


@st.cache_data(show_spinner=False)
def monte_carlo(Lt, b, h, L, dx, ny, m, sig_th, n, graine):
    g = grille(Lt, b, h, L, dx, ny)
    F_ref = 2 * b * h**2 * sig_th / (3 * L)
    sigma_L = calibrer(g, m, F_ref)
    F, idx = simuler(g, m, sigma_L, n, graine)
    return F, g["X"][g["masque"]][idx], g["Y"][g["masque"]][idx]


# --------------------------------------------------------------------------------------
# Graphiques
# --------------------------------------------------------------------------------------
def carte(g, Z, titre, cmap, vmin, vmax, etiquette, crit=None):
    fig, ax = plt.subplots(figsize=(11, 2.8))
    im = ax.imshow(Z, extent=[0, g["Lt"], g["h"], 0], aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    xs = [g["xc"] - g["L"] / 2, g["xc"] + g["L"] / 2]
    ax.plot(xs, [g["h"]] * 2, "^", color="k", ms=10, clip_on=False)
    ax.plot([g["xc"]], [0], "v", color="k", ms=10, clip_on=False)
    if crit is not None:
        ax.plot(*crit, marker="x", color="white", mec="k", ms=12, mew=2)
    ax.set_xlabel("Position longitudinale x (mm)")
    ax.set_ylabel("Profondeur y (mm)")
    ax.set_title(titre)
    fig.colorbar(im, ax=ax, label=etiquette, pad=0.01)
    fig.tight_layout()
    return fig


def papier_weibull(F, m_vrai=None):
    F = np.sort(F)
    P = (np.arange(1, F.size + 1) - 0.5) / F.size
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(np.log(F), np.log(-np.log1p(-P)), ".", ms=2, label="Simulations")
    m_e, F63 = ajuster_regression(F)
    xx = np.linspace(np.log(F[0]), np.log(F[-1]), 50)
    ax.plot(xx, m_e * (xx - np.log(F63)), "r-", label=f"Droite ajustée (m = {m_e:.1f})")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("ln F")
    ax.set_ylabel("ln ln [1 / (1 − P)]")
    ax.set_title("Papier de Weibull")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Application Streamlit
# --------------------------------------------------------------------------------------
PRESETS = {
    "Notre projet (120 × 15 × 10 mm, portée 100 mm)":
        dict(Lt=120.0, b=15.0, h=10.0, L=100.0, sig=20.0, m=10.0),
    "Illustration du professeur (150 × 10 × 5 mm, portée 85 mm)":
        dict(Lt=150.0, b=10.0, h=5.0, L=85.0, sig=20.0, m=13.0),
}


def main():
    st.set_page_config(page_title="Weibull : flexion 3 points", layout="wide")
    st.title("Modèle de Weibull : simulation Monte-Carlo d'un essai de flexion 3 points")
    st.caption("Répartition aléatoire des résistances locales dans une céramique "
               "(faïence) et effet sur la force à rupture.")

    with st.sidebar:
        st.header("Paramètres")
        noms = list(PRESETS)
        nom = st.selectbox("Configuration de départ", noms)
        p = PRESETS[nom]
        k = noms.index(nom)  # change les clés des widgets quand on change de configuration
        st.subheader("Éprouvette et essai")
        Lt = st.number_input("Longueur totale (mm)", 10.0, 300.0, p["Lt"], 1.0, key=f"Lt{k}")
        b = st.number_input("Largeur b (mm)", 1.0, 30.0, p["b"], 0.5, key=f"b{k}")
        h = st.number_input("Épaisseur h (mm)", 1.0, 20.0, p["h"], 0.5, key=f"h{k}")
        L = st.number_input("Portée L entre appuis (mm)", 5.0, 280.0, p["L"], 1.0, key=f"L{k}")
        st.subheader("Matériau")
        sig_th = st.number_input("σ caractéristique de l'éprouvette (MPa)", 1.0, 200.0,
                                 p["sig"], 1.0, key=f"sig{k}",
                                 help="Contrainte à rupture à 63,2 % de probabilité de rupture. "
                                      "Valeur illustrative, à remplacer par vos mesures.")
        m = st.slider("Module de Weibull m", 2.0, 30.0, p["m"], 0.5, key=f"m{k}")
        st.subheader("Simulation")
        dx = st.number_input("Taille de maille en x (mm)", 0.25, 5.0, 1.0, 0.25, key=f"dx{k}")
        ny = st.number_input("Nombre de mailles en profondeur", 6, 80, 30, 2, key=f"ny{k}")
        n_sim = st.select_slider("Nombre de simulations", [1000, 2000, 5000, 10000, 20000, 50000],
                                 value=10000, key=f"n{k}")
        graine = st.number_input("Graine aléatoire", 0, 10**6, 1, 1, key=f"g{k}")

    if L >= Lt:
        st.error("La portée doit être inférieure à la longueur de l'éprouvette.")
        st.stop()
    if L / h < 10:
        st.warning(f"L/h = {L / h:.1f} < 10 : la formule de flexion n'est plus valable "
                   "(cisaillement non négligeable). Le cahier des charges demande 10 à 16.")

    g = grille(Lt, b, h, L, dx, int(ny))
    F_ref = 2 * b * h**2 * sig_th / (3 * L)
    sigma_L = calibrer(g, m, F_ref)

    with st.expander("Hypothèses du modèle (à lire avant d'interpréter les résultats)"):
        st.markdown(
            "- **Maillon le plus faible** : l'éprouvette casse dès qu'une seule maille atteint sa "
            "résistance locale.\n"
            "- **Mailles indépendantes** : pas de longueur de corrélation entre défauts voisins.\n"
            "- **Une seule population de défauts** (volume). Les défauts de surface, d'arête "
            "et d'usinage ne sont pas modélisés séparément.\n"
            "- **Poutre élastique linéaire** : contrainte donnée par la théorie des poutres, "
            "sans effet du cisaillement ni des contacts sous les appuis et le poinçon.\n"
            "- **Le volume compte** : la loi locale est ajustée au volume de chaque maille, donc "
            "le résultat ne dépend presque pas de la finesse du maillage.\n"
            "- **Calibrage** : σ caractéristique est celle de l'éprouvette entière, pour la "
            "portée choisie. Les autres portées de l'onglet 4 utilisent la même résistance locale.\n"
            "- Valeurs de σ et m **illustratives** : à remplacer par celles issues de vos essais.")

    t1, t2, t3, t4 = st.tabs(["1. Une éprouvette", "2. Monte-Carlo",
                              "3. Weibull et nombre d'éprouvettes", "4. Effet de la portée"])

    # ---------------------------- onglet 1 ----------------------------
    with t1:
        if "compteur" not in st.session_state:
            st.session_state.compteur = 0
        if st.button("Tirer une nouvelle éprouvette"):
            st.session_state.compteur += 1
        e = eprouvette_unique(g, m, sigma_L, int(graine) * 1000 + st.session_state.compteur)
        sig_nom = 3 * e["F"] * L / (2 * b * h**2)
        c1, c2, c3 = st.columns(3)
        c1.metric("Force théorique RDM (sans défauts)", f"{F_ref:.2f} N")
        c2.metric("Force simulée à rupture", f"{e['F']:.2f} N")
        c3.metric("Point critique (x ; y)", f"{e['xc_crit']:.1f} ; {e['yc_crit']:.2f} mm")
        st.caption(f"Référence RDM : F = 2 b h² σ / (3 L) avec σ = {sig_th:.0f} MPa. "
                   f"Contrainte nominale simulée à rupture : {sig_nom:.2f} MPa. "
                   "La portée et le chargement modifient les contraintes, pas les défauts. "
                   "Une nouvelle éprouvette tire un autre champ aléatoire.")
        crit = (e["xc_crit"], e["yc_crit"])
        indice = 1 - e["U"]  # 1 = maille faible (rouge), 0 = maille résistante (bleu)
        st.pyplot(carte(g, indice, "Carte des défauts (bleu : résistant ; rouge : faible)",
                        "coolwarm", 0, 1, "Indice relatif de défaut (sans unité)", crit))
        charge = st.slider("Chargement appliqué (% de la force de rupture)", 0, 100, 100)
        F_app = e["F"] * charge / 100
        st.pyplot(carte(g, F_app * g["s_signe"], f"Contrainte longitudinale à F = {F_app:.2f} N "
                        "(traction > 0)", "RdBu_r", -F_app * g["s_signe"].max() - 1e-9,
                        F_app * g["s_signe"].max() + 1e-9, "σx (MPa)", crit))
        util = np.where(g["masque"], F_app * g["s_signe"] / e["sig_loc"], np.nan)
        st.pyplot(carte(g, util, "Taux de sollicitation σ / σ locale à rupture "
                        "(1 = rupture de la maille)", "YlOrRd", 0, 1, "σ / σ locale", crit))

    # ---------------------------- onglet 2 ----------------------------
    F, xs, ys = monte_carlo(Lt, b, h, L, dx, int(ny), m, sig_th, int(n_sim), int(graine))
    m_r, F63_r = ajuster_regression(F)
    with t2:
        c = st.columns(4)
        c[0].metric("Simulations", f"{len(F):,}".replace(",", " "))
        c[1].metric("Force moyenne", f"{F.mean():.2f} N")
        c[2].metric("Force de référence RDM", f"{F_ref:.2f} N")
        c[3].metric("Force caractéristique (ajustement)", f"{F63_r:.2f} N")
        c = st.columns(3)
        c[0].metric("Rapport moyen F / F_ref", f"{F.mean() / F_ref:.3f}")
        c[1].metric("Rapport caractéristique (ajustement)", f"{F63_r / F_ref:.3f}")
        c[2].metric("Écart type / moyenne", f"{F.std(ddof=1) / F.mean():.3f}")
        t = F / F_ref
        fig, ax = plt.subplots(figsize=(11, 4.2))
        bornes = np.linspace(t.min(), t.max(), 81)
        ax.hist(t, bins=bornes, color="#6fa8c9", edgecolor="white", label="Nombre d'apparitions (Monte-Carlo)")
        tt = np.linspace(t.min(), t.max(), 400)
        ax.plot(tt, len(t) * (bornes[1] - bornes[0]) * pdf_weibull(tt, m, 1.0), "r-", lw=2,
                label=f"Weibull théorique (m = {m:.1f}, effectifs attendus)")
        ax.axvline(1, color="darkgreen", ls="--", lw=2, label="Référence RDM : rapport = 1")
        ax.set_xlabel("Force normalisée F_rupture / F_ref (sans unité)")
        ax.set_ylabel("Nombre d'apparitions")
        ax.set_title("Répartition des forces de rupture normalisées")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("**Où la rupture s'amorce** (toutes les simulations)")
        fig, ax = plt.subplots(figsize=(11, 3))
        hh = ax.hist2d(xs, ys, bins=[min(60, g["nx"]), g["ny"] // 2], cmap="magma_r",
                       range=[[0, Lt], [h / 2, h]])
        ax.set_xlim(0, Lt)
        ax.invert_yaxis()
        ax.set_xlabel("Position longitudinale x (mm)")
        ax.set_ylabel("Profondeur y (mm)")
        ax.set_title("Densité des points d'amorçage (moitié tendue de l'éprouvette)")
        ax.plot([g["xc"] - L / 2, g["xc"] + L / 2], [h, h], "^", color="k", ms=9, clip_on=False)
        fig.colorbar(hh[3], ax=ax, label="Nombre de ruptures amorcées")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(f"{100 * np.mean(ys > 0.9 * h):.0f} % des ruptures s'amorcent dans les 10 % "
                   "d'épaisseur les plus proches de la face tendue, et "
                   f"{100 * np.mean(np.abs(xs - g['xc']) < L / 4):.0f} % dans la moitié centrale entre appuis.")

    # ---------------------------- onglet 3 ----------------------------
    with t3:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.pyplot(papier_weibull(F))
        with c2:
            m_ml, F63_ml = mle(F)
            st.metric("m imposé dans le modèle", f"{m:.2f}")
            st.metric("m estimé par régression (papier de Weibull)", f"{m_r:.2f}")
            st.metric("m estimé par maximum de vraisemblance", f"{m_ml:.2f}")
            st.caption("Avec beaucoup d'éprouvettes, les deux méthodes retrouvent le m imposé. "
                       "Avec peu d'éprouvettes, ce n'est plus le cas : voir ci-dessous.")
        st.subheader("Combien d'éprouvettes pour estimer m ?")
        st.caption("On tire des lots de n éprouvettes dans les simulations (tirage avec remise), "
                   "on estime m sur chaque lot, et on regarde la dispersion de l'estimation.")
        tailles = [5, 10, 15, 20, 30, 50]
        rng = np.random.default_rng(int(graine) + 7)
        K = 2000
        res_reg, res_mle = [], []
        for n in tailles:
            Fs = F[rng.integers(0, F.size, (K, n))]
            res_reg.append(regression_lignes(Fs))
            res_mle.append(mle_lignes(Fs))
        fig, ax = plt.subplots(figsize=(11, 4))
        pos = np.arange(len(tailles))
        b1 = ax.boxplot(res_reg, positions=pos - 0.18, widths=0.3, patch_artist=True, showfliers=False)
        b2 = ax.boxplot(res_mle, positions=pos + 0.18, widths=0.3, patch_artist=True, showfliers=False)
        for bx in b1["boxes"]:
            bx.set_facecolor("#9ecae1")
        for bx in b2["boxes"]:
            bx.set_facecolor("#fdae6b")
        ax.axhline(m, color="red", ls="--", label=f"m imposé = {m:.1f}")
        ax.set_xticks(pos)
        ax.set_xticklabels(tailles)
        ax.set_xlabel("Nombre d'éprouvettes par lot")
        ax.set_ylabel("m estimé")
        ax.legend(handles=[b1["boxes"][0], b2["boxes"][0], ax.lines[-1]],
                  labels=["Régression", "Maximum de vraisemblance", f"m imposé = {m:.1f}"])
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        tab = {"n": tailles,
               "m médian (régression)": [round(float(np.median(r)), 2) for r in res_reg],
               "intervalle 5-95 % (régression)": [f"{np.percentile(r, 5):.1f} à {np.percentile(r, 95):.1f}" for r in res_reg],
               "m médian (vraisemblance)": [round(float(np.median(r)), 2) for r in res_mle],
               "intervalle 5-95 % (vraisemblance)": [f"{np.percentile(r, 5):.1f} à {np.percentile(r, 95):.1f}" for r in res_mle]}
        st.dataframe(tab, hide_index=True)

    # ---------------------------- onglet 4 ----------------------------
    with t4:
        st.caption("Même matériau (même résistance locale), portées différentes. "
                   "Une portée plus longue sollicite plus de volume en traction, donc "
                   "davantage de défauts.")
        portees = sorted({round(r * h, 1) for r in (4, 6, 8, 10, 12, 16) if r * h <= Lt - 4} | {L})
        lignes = []
        for s in portees:
            gs = grille(Lt, b, h, float(s), dx, int(ny))
            Fc = sigma_L / (np.sum(gs["V"] / V0 * gs["s_t"] ** m)) ** (1 / m)
            Fs, _ = simuler(gs, m, sigma_L, 3000, int(graine) + int(s))
            fac = 3 * s / (2 * b * h**2)
            lignes.append({"Portée L (mm)": s, "L/h": round(s / h, 1),
                           "F caractéristique (N)": round(float(Fc), 2),
                           "σ nominale caractéristique (MPa)": round(float(fac * Fc), 2),
                           "σ nominale moyenne (MPa)": round(float(fac * Fs.mean()), 2),
                           "Cisaillement négligeable (L/h ≥ 10)": "oui" if s / h >= 10 else "non"})
        st.dataframe(lignes, hide_index=True)
        st.caption("La contrainte nominale diminue quand la portée augmente : c'est l'effet de volume. "
                   "Pour comparer des lots, il faut donc garder la même géométrie et la même portée. "
                   "En dessous de L/h = 10, le cisaillement fausse la formule de flexion.")


if __name__ == "__main__":
    main()
