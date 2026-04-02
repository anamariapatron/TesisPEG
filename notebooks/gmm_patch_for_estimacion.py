#librerias
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import os

import contextlib
import joblib
from tqdm.notebook import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

        
def average_hourly_terms_gmm(
    df_bootstrap_list,
    df_transado,
    df_full,
    fecha_hora,
    gamma,
    firma,
    compute_equilibrium,
    kernel_expectation,
    kernel_derivative_weighted,
    price_tolerance=1e-8,
):
    """
    Construye los objetos horarios necesarios para el GMM.

    Devuelve promedios bootstrap de:
      - E[Q^{bs}_{iht}]
      - E[dQ/db]
      - E[1{dp/db = 1}]
      - A_iht = E[ I * ((Q-pos) + b*dQdb) ]
      - B_iht = E[ I * dQdb ]

    Nota: si la subasta es hora a hora, se interpreta b = b_iht y dQ/db = dQ_iht/db_iht.
    """
    if not df_bootstrap_list:
        return {
            'EQ_bs': np.nan,
            'EdQdb_bs': np.nan,
            'EIprice_bs': np.nan,
            'A_iht': np.nan,
            'B_iht': np.nan,
            'n_valid_boot': 0,
        }

    df_firma = df_full[
        (df_full['CodigoPlanta'] == firma) &
        (df_full['FechaHora'] == fecha_hora)
    ].copy()

    if df_firma.empty:
        return {
            'EQ_bs': np.nan,
            'EdQdb_bs': np.nan,
            'EIprice_bs': np.nan,
            'A_iht': np.nan,
            'B_iht': np.nan,
            'n_valid_boot': 0,
        }

    bit = float(df_firma['precio'].iloc[0])
    pos_iht = float(df_firma['cantidad_pos'].iloc[0])

    Q_vals = []
    dQ_vals = []
    I_vals = []
    A_vals = []
    B_vals = []

    for df_sim in df_bootstrap_list:
        # asegurar fecha de referencia para compute_equilibrium
        df_firma_bs = df_firma.copy()
        df_firma_bs['fecha_base'] = fecha_hora
        df_sim_bs = df_sim.copy()
        if 'fecha_base' not in df_sim_bs.columns:
            df_sim_bs['fecha_base'] = fecha_hora

        df_bidders = pd.concat([df_sim_bs, df_firma_bs], ignore_index=True)
        p_star, _ = compute_equilibrium(df_bidders, df_transado)

        if pd.isna(p_star):
            continue

        demand_row = df_transado.loc[df_transado['FechaHora'] == fecha_hora, 'demanda']
        if demand_row.empty:
            continue
        D = demand_row.iloc[0]

        Q_bs = kernel_expectation(df_sim_bs, p_star, D, gamma, df_firma_bs)
        dQ_db_bs = kernel_derivative_weighted(df_sim_bs, p_star, gamma, df_firma_bs)

        if pd.isna(Q_bs) or pd.isna(dQ_db_bs):
            continue

        # Indicador de si la firma fija el precio en este bootstrap
        I_price_bs = float(np.isclose(bit, p_star, atol=price_tolerance))

        A_bs = I_price_bs * ((Q_bs - pos_iht) + bit * dQ_db_bs)
        B_bs = I_price_bs * dQ_db_bs

        Q_vals.append(float(Q_bs))
        dQ_vals.append(float(dQ_db_bs))
        I_vals.append(I_price_bs)
        A_vals.append(float(A_bs))
        B_vals.append(float(B_bs))

    if len(A_vals) == 0:
        return {
            'EQ_bs': np.nan,
            'EdQdb_bs': np.nan,
            'EIprice_bs': np.nan,
            'A_iht': np.nan,
            'B_iht': np.nan,
            'n_valid_boot': 0,
        }

    return {
        'EQ_bs': np.mean(Q_vals),
        'EdQdb_bs': np.mean(dQ_vals),
        'EIprice_bs': np.mean(I_vals),
        'A_iht': np.mean(A_vals),
        'B_iht': np.mean(B_vals),
        'n_valid_boot': len(A_vals),
    }



def process_row_gmm(
    row,
    df_full,
    df_transado,
    gamma,
    M,
    get_similar_days_by_cluster,
    bootstrap_by_planta,
    compute_equilibrium,
    kernel_expectation,
    kernel_derivative_weighted,
    bootstrap_seed=123,
    price_tolerance=1e-8,
):
    fecha_hora = row.FechaHora
    firma = row.CodigoPlanta

    df_similares = get_similar_days_by_cluster(df_full, fecha_hora, firma)
    if df_similares.empty:
        return {
            'EQ_bs': np.nan,
            'EdQdb_bs': np.nan,
            'EIprice_bs': np.nan,
            'A_iht': np.nan,
            'B_iht': np.nan,
            'n_valid_boot': 0,
        }

    df_bootstrap_list = bootstrap_by_planta(df_similares, M, seed=bootstrap_seed)

    return average_hourly_terms_gmm(
        df_bootstrap_list=df_bootstrap_list,
        df_transado=df_transado,
        df_full=df_full,
        fecha_hora=fecha_hora,
        gamma=gamma,
        firma=firma,
        compute_equilibrium=compute_equilibrium,
        kernel_expectation=kernel_expectation,
        kernel_derivative_weighted=kernel_derivative_weighted,
        price_tolerance=price_tolerance,
    )




    
def build_hourly_gmm_terms_parallel(
    df,
    df_transado,
    gamma,
    M,
    get_similar_days_by_cluster,
    bootstrap_by_planta,
    compute_equilibrium,
    kernel_expectation,
    kernel_derivative_weighted,
    bootstrap_seed=123,
    price_tolerance=1e-8,
    n_jobs=-1,
):
    """Corre el bootstrap/kernel fila a fila (planta-hora) y devuelve objetos horarios para GMM."""
    rows = [row for _, row in df.iterrows()]

    with tqdm_joblib(tqdm(desc="Procesando horas", total=len(rows))):
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_row_gmm)(
                row=row,
                df_full=df,
                df_transado=df_transado,
                gamma=gamma,
                M=M,
                get_similar_days_by_cluster=get_similar_days_by_cluster,
                bootstrap_by_planta=bootstrap_by_planta,
                compute_equilibrium=compute_equilibrium,
                kernel_expectation=kernel_expectation,
                kernel_derivative_weighted=kernel_derivative_weighted,
                bootstrap_seed=bootstrap_seed,
                price_tolerance=price_tolerance,
            )
            for row in rows
        )
    out = df.copy().reset_index(drop=True)
    out['EQ_bs'] = [r['EQ_bs'] for r in results]
    out['EdQdb_bs'] = [r['EdQdb_bs'] for r in results]
    out['EIprice_bs'] = [r['EIprice_bs'] for r in results]
    out['A_iht'] = [r['A_iht'] for r in results]
    out['B_iht'] = [r['B_iht'] for r in results]
    out['n_valid_boot'] = [r['n_valid_boot'] for r in results]
    return out


def _compute_S1_S2(group, q_min):
    """Construye los agregados diarios S1 y S2 a partir de cantidades observadas hora a hora."""
    g = group.sort_values('FechaHora').copy()
    q = g['cantidad'].astype(float).to_numpy()

    q_prev = np.roll(q, 1)
    q_next = np.roll(q, -1)
    q_prev[0] = q[0]
    q_next[-1] = q[-1]

    s1 = np.maximum(q - q_min, 0.0).sum()
    s2 = (2.0 * q - q_prev - q_next).sum()
    return s1, s2



def build_daily_panel_for_gmm(
    hourly_df,
    q_min_map=None,
    foil6_col='COMBUSTOLEO',
):
    """
    Agrega los términos horarios a nivel planta-día y construye x_it.

    q_min_map: dict opcional {CodigoPlanta: underline_Q_i}. Si no se pasa,
    se usa el mínimo observado positivo de `cantidad` por planta.
    """
    df = hourly_df.copy()
    df['FechaHora'] = pd.to_datetime(df['FechaHora'], errors='coerce')
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

    if q_min_map is None:
        tmp = df.loc[df['cantidad'] > 0].groupby('CodigoPlanta')['cantidad'].min()
        q_min_map = tmp.to_dict()

    daily_rows = []
    for (planta, fecha), g in df.groupby(['CodigoPlanta', 'Fecha'], dropna=False):
        g = g.sort_values('FechaHora').copy()

        q_min = float(q_min_map.get(planta, 0.0))
        s1, s2 = _compute_S1_S2(g, q_min)

        # A_it y B_it son sumas sobre horas
        A_it = g['A_iht'].sum(min_count=1)
        B_it = g['B_iht'].sum(min_count=1)

        # shifters diarios: como están replicados en cada hora, tomamos el primero del día
        first = g.iloc[0]

        pfuel = float(first['pfuel']) if pd.notna(first['pfuel']) else np.nan
        trm = float(first['trm']) if pd.notna(first['trm']) else np.nan
        cere = float(first['CERE']) if pd.notna(first['CERE']) else np.nan
        fazni = float(first['FAZNI']) if pd.notna(first['FAZNI']) else np.nan
        foil6 = float(first[foil6_col]) if foil6_col in first.index and pd.notna(first[foil6_col]) else np.nan

        daily_rows.append({
            'CodigoPlanta': planta,
            'Fecha': fecha,
            'A_it': A_it,
            'B_it': B_it,
            'S1_it': s1,
            'S2_it': s2,
            'pfuel': pfuel,
            'foil6': foil6,
            'trm': trm,
            'CERE': cere,
            'FAZNI': fazni,
            'dow': fecha.dayofweek if pd.notna(fecha) else np.nan,
            'month': fecha.month if pd.notna(fecha) else np.nan,
            'n_hours': len(g),
            'mean_EIprice_bs': g['EIprice_bs'].mean(),
        })

    daily = pd.DataFrame(daily_rows)

    # logs seguros
    for c in ['pfuel', 'foil6', 'trm', 'CERE', 'FAZNI']:
        daily[f'ln_{c}'] = np.where(daily[c] > 0, np.log(daily[c]), np.nan)

    # x_it (14 parámetros)
    daily['x_1'] = 1.0
    daily['x_2'] = daily['ln_pfuel']
    daily['x_3'] = daily['ln_foil6']
    daily['x_4'] = daily['ln_trm']
    daily['x_5'] = daily['ln_CERE']
    daily['x_6'] = daily['ln_FAZNI']
    daily['x_7'] = daily['S1_it']
    daily['x_8'] = daily['S1_it'] * daily['ln_pfuel']
    daily['x_9'] = daily['S1_it'] * daily['ln_foil6']
    daily['x_10'] = daily['S1_it'] * daily['ln_trm']
    daily['x_11'] = daily['S2_it']
    daily['x_12'] = daily['S2_it'] * daily['ln_pfuel']
    daily['x_13'] = daily['S2_it'] * daily['ln_foil6']
    daily['x_14'] = daily['S2_it'] * daily['ln_trm']

    return daily



def add_basic_instruments(daily_df):
    """Instrumentos básicos diarios. Puedes ampliarlos después."""
    df = daily_df.copy()

    # dummies de día de semana y mes
    dow_dummies = pd.get_dummies(df['dow'], prefix='dow', drop_first=True, dtype=float)
    month_dummies = pd.get_dummies(df['month'], prefix='m', drop_first=True, dtype=float)

    Z = pd.concat([
        pd.Series(1.0, index=df.index, name='z_const'),
        df[['ln_pfuel', 'ln_foil6', 'ln_trm', 'ln_CERE', 'ln_FAZNI']].astype(float),
        dow_dummies,
        month_dummies,
    ], axis=1)

    Z.columns = [c.replace('ln_', 'z_') if c.startswith('ln_') else c for c in Z.columns]
    out = pd.concat([df, Z], axis=1)
    return out, Z.columns.tolist()



def estimate_xi_identity_gmm(
    daily_df,
    plant,
    x_cols=None,
    z_cols=None,
    ridge=1e-8,
    dropna=True,
):
    """
    Estima xi_i por GMM lineal con ponderación identidad:
        m_it(xi) = A_it - B_it * x_it' xi
    """
    if x_cols is None:
        x_cols = [f'x_{k}' for k in range(1, 15)]

    dfp = daily_df.loc[daily_df['CodigoPlanta'] == plant].copy()

    if z_cols is None:
        z_cols = [c for c in dfp.columns if c.startswith('z_') or c.startswith('dow_') or c.startswith('m_')]

    keep_cols = ['A_it', 'B_it'] + x_cols + z_cols
    if dropna:
        dfp = dfp.dropna(subset=keep_cols).copy()

    if dfp.empty:
        raise ValueError(f'No hay observaciones válidas para la planta {plant}.')

    A = dfp['A_it'].to_numpy(dtype=float)[:, None]      # T x 1
    B = dfp['B_it'].to_numpy(dtype=float)[:, None]      # T x 1
    X = dfp[x_cols].to_numpy(dtype=float)               # T x K
    Z = dfp[z_cols].to_numpy(dtype=float)               # T x L

    G = B * X                                           # T x K

    ZtG = Z.T @ G                                       # L x K
    ZtA = Z.T @ A                                       # L x 1

    M1 = G.T @ Z @ ZtG                                  # K x K
    M2 = G.T @ Z @ ZtA                                  # K x 1

    xi_hat = np.linalg.pinv(M1 + ridge * np.eye(M1.shape[0])) @ M2
    xi_hat = xi_hat.flatten()

    dfp['phi_hat'] = X @ xi_hat
    dfp['m_hat'] = (A.flatten() - np.sum(G * xi_hat[None, :], axis=1))

    coef_names = [
        'phi_cons_1', 'phi_Pfuel_1', 'phi_Foil6_1', 'phi_TRM_1', 'phi_CERE_1', 'phi_FAZN_1',
        'phi_cons_2', 'phi_Pfuel_2', 'phi_Foil6_2', 'phi_TRM_2',
        'phi_cons_3', 'phi_Pfuel_3', 'phi_Foil6_3', 'phi_TRM_3',
    ]
    coef = pd.Series(xi_hat, index=coef_names, name=plant)

    return {
        'plant': plant,
        'coef': coef,
        'daily_fit': dfp,
        'x_cols': x_cols,
        'z_cols': z_cols,
        'n_obs': len(dfp),
    }

##############################################################
################# old functions ##############################
# ----------------------------------------------------
# FUNCIONES AUXILIARES
# ----------------------------------------------------

def get_cluster_it(df, fecha, firma):
    """Devuelve el cluster correspondiente a una firma i en una FechaHora t."""
    row = df.loc[
        (df['FechaHora'] == fecha) & (df['CodigoPlanta'] == firma),
        'cluster'
    ]
    return row.iloc[0] if not row.empty else np.nan


def get_competitors(df, fecha, firma):
    """Devuelve los competidores (otros CodigoPlanta) presentes en la misma FechaHora."""
    df_day = df[df['FechaHora'] == fecha]
    competitors = df_day.loc[df_day['CodigoPlanta'] != firma, 'CodigoPlanta'].unique().tolist()
    return competitors


def get_similar_days_by_cluster(df, fecha, firma, max_obs=20):

    cluster_it = get_cluster_it(df, fecha, firma)
    competitors = get_competitors(df, fecha, firma)

    # cantidad de la firma en esa fecha
    cantidad_i = df.loc[(df['CodigoPlanta'] == firma) & (df['FechaHora'] == fecha), 'cantidad'].item()
    
    
    similar_obs = []

    for comp in competitors:

        df_comp_similar = df[
            (df['CodigoPlanta'] == comp) &
            (df['cluster'] == cluster_it)
        ].copy()

        # calcular distancia en cantidad a la firma
        df_comp_similar['dist_cantidad'] = (
            df_comp_similar['cantidad'] - cantidad_i
        ).abs()

        # ordenar por cercanía y tomar máximo max_obs
        df_comp_similar = (
            df_comp_similar
            .sort_values('dist_cantidad')
            .head(max_obs)
        )

        df_comp_similar['competidor_de'] = firma
        df_comp_similar['fecha_base'] = fecha

        similar_obs.append(df_comp_similar)

    return pd.concat(similar_obs, ignore_index=True) if similar_obs else pd.DataFrame()


def bootstrap_by_planta(df, M, seed=None):
    """Genera M muestras bootstrap independientes seleccionando 1 observación por planta."""
    if seed is not None:
        np.random.seed(seed)

    plantas = df['CodigoPlanta'].unique()
    bootstrap_samples = []

    for m in range(M):
        muestras = []
        for p in plantas:
            df_p = df[df['CodigoPlanta'] == p]
            if len(df_p) == 0:
                continue
            muestra = df_p.sample(1, replace=True)
            muestra['bootstrap_id'] = m + 1
            muestras.append(muestra)
        sample_df = pd.concat(muestras).reset_index(drop=True)
        bootstrap_samples.append(sample_df)

    return bootstrap_samples


def compute_equilibrium(df_offers, df_transado_date):
    """
    Encuentra el precio y cantidad de equilibrio (p*, q*) para un conjunto de ofertas.
    df_transado_date debe contener la demanda para la FechaHora actual.
    """
    df_transado_date['FechaHora'] = pd.to_datetime(df_transado_date['FechaHora'], errors='coerce')
    df_offers['fecha_base'] = pd.to_datetime(df_offers['fecha_base'], errors='coerce')

    fecha = df_offers['fecha_base'].iloc[0]
   

    demanda_row = df_transado_date.loc[df_transado_date['FechaHora'] == fecha, 'demanda']
    
    
    if demanda_row.empty:
        return np.nan, np.nan

    demand = demanda_row.iloc[0]
    df_sorted = df_offers.sort_values('precio').copy()
    df_sorted['acum'] = df_sorted['cantidad_pos'].cumsum()

    clearing_offers = df_sorted[df_sorted['acum'] >= demand]
    if clearing_offers.empty:
        return np.nan, demand

    p_star = clearing_offers.iloc[0]['precio']
    q_star = demand
    
    return p_star, q_star

# ----------------------------------------------------
# KERNELS
# ----------------------------------------------------
# --- 1. Definiciones del Kernel ---

def gaussian_kernel(u):
    """Kernel Gaussiano estándar (PDF de N(0, 1))."""
    return norm.pdf(u)

def gaussian_kernel_prime(u):
    """Derivada del Kernel Gaussiano: κ'(u) = -u * κ(u)."""
    return -u * gaussian_kernel(u)   #ya es como si tuviera el negativo, u lo reescribo como  pht-pkt por justificacion

# --- 2. Estimación de la Demanda Residual (RD(p)) ---

# Asumo que tienes una función para obtener pos_it, o que se añade como argumento
def kernel_expectation(df, p_ht, D, gamma, df_firma):
    """
    Calcula la Demanda Residual Neta:
       RD(p) = D - sum_k g_k * K((b_k - p)/gamma)
    """
    D = float(np.asarray(D).squeeze())

    if df.empty:
        return D

    gammai = gamma * float(df_firma['gamma_thumb'].iloc[0])

    # Si quieres evaluar en p_ht, debe ir p_ht aquí.
    u_others = (df["precio"] - p_ht) / gammai

    weights_others = gaussian_kernel(u_others)

    weight = (df["precio"] < p_ht).astype(float)

    S_minus_i = (df['cantidad_pos'] * weights_others * weight).sum()

    RD_p = D - float(S_minus_i)

    q_min = float(df_firma['cantidad_pos'].iloc[0])
    resultado = min(RD_p, q_min)

    return resultado


# --- 3. Estimación de la Derivada de la Demanda Residual (RD'(p)) ---

def kernel_derivative(df, p_ht, gamma, df_firma):
    """
    Calcula la derivada de la Demanda Residual:
        RD'(p) = sum_{k ≠ i} g_k * (1/gamma) * K'((b_k - p)/gamma)
    """
    if df.empty:
        return 0.0
    gammai=gamma* df_firma['gamma_thumb'].values[0]
    # Coincidir EXACTAMENTE con tu fórmula: (b_k - p_ht)/gamma
    u_others = (df["precio"] - p_ht) / gammai

    # K'(u) = -u K(u)
    weights_prime = gaussian_kernel_prime(u_others)

    # SUMA ponderada (no promedio)
    dQ = (df["cantidad"] * weights_prime).sum() / gammai
    
    return dQ




def kernel_derivative_weighted(df, p_ht, gamma, df_firma):
    """
    Calcula la derivada de la Demanda Residual ponderando más los bids que no entran
    RD'(p) = sum_{k ≠ i} g_k * (1/gamma) * K'((b_k - p)/gamma) * w(b_k, p_ht)
    """
    if df.empty:
        return 0.0
    bit=df_firma['precio'].values[0]
    gammai=gamma*df_firma['gamma_thumb'].values[0]
    # Vector u = (b_k - p_h)/gamma
    u_others = (df["precio"] - p_ht) / gammai

    # derivada del kernel normal: K'(u) = -u K(u)
    weights_prime = gaussian_kernel_prime(u_others)

    # Ponderación: más peso si b_k > p_h
    # opción 1: binaria
    weight = (df["precio"] > p_ht).astype(float)
    # opción 2: continua usando sigmoid
    beta=5
    #weight = 1 / (1 + np.exp(-beta * (df["precio"] - p_ht)))

    # SUMA ponderada
    dQ = (df["cantidad"] * weights_prime * weight).sum() / gammai
    
    return dQ

#otro u_others = (df["precio"] - p_ht) / gammai