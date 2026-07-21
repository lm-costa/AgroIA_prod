import os
import glob
import re
import warnings
import xarray as xr
import numpy as np
import pandas as pd
import gc

warnings.simplefilter("ignore")

print("=== FÁBRICA DE HARMONIZAÇÃO (MODO DE ADAPTAÇÃO DINÂMICA DE CALENDÁRIO) ===")

# 1. PARÂMETROS GERAIS
CENARIOS = ['ssp126', 'ssp245', 'ssp585']
MODELOS = ['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'UKESM1-0-LL']
VARIAVEIS = ['tasmax', 'tasmin', 'pr', 'rsds', 'sfcWind', 'hurs']

PERIODO_HIST = slice('1985', '2014')
PERIODO_OVERLAP = slice('2015', '2025')
ANOS_OVERLAP = range(2015, 2026)

# 2. DIRETÓRIOS LOCAIS E GOOGLE DRIVE DESKTOP
BASE_ERA5 = r'D:\ERA5'
BASE_NEX = r'G:\Drives compartilhados\GAS-Henrique\NEX-GDDP-CMIP6'
BASE_SAIDA_NC = r'G:\Drives compartilhados\GAS-Henrique\NEX_CORRIGIDO_TESTE'
DIR_SAIDA_TABELAS = r'G:\Drives compartilhados\GAS-Henrique\Tabelas_Artigo'

os.makedirs(DIR_SAIDA_TABELAS, exist_ok=True)


# 3. FUNÇÕES AUXILIARES
def carregar_dados(pasta, ano_inicio=None, ano_fim=None):
    if not os.path.exists(pasta): return None
    arquivos_todos = sorted(glob.glob(os.path.join(pasta, "*.nc4")))

    arquivos_filtrados = []
    for arq in arquivos_todos:
        match = re.search(r'_(\d{4})\.nc4$', arq)
        if match:
            ano = int(match.group(1))
            if ano_inicio and ano < ano_inicio: continue
            if ano_fim and ano > ano_fim: continue
        arquivos_filtrados.append(arq)

    if not arquivos_filtrados: return None

    try:
        # LEITURA NATIVA PURA E ULTRA-RÁPIDA (O Python não traduz, apenas empilha)
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_mfdataset(
            arquivos_filtrados,
            engine='netcdf4',
            chunks={'time': 365, 'lat': 50, 'lon': 50},
            decode_times=time_coder,
            combine='nested',
            concat_dim='time',
            compat='override',
            coords='minimal'
        )
        # Remove sobreposições de matriz
        _, index = np.unique(ds['time'], return_index=True)
        ds = ds.isel(time=index)
        return ds
    except Exception as e:
        print(f"      [Erro Crítico] Falha ao ler {pasta}. Erro: {e}")
        return None


def calcular_metricas_avancadas(obs_da, mod_da):
    obs_flat = obs_da.values.flatten()
    mod_flat = mod_da.values.flatten()

    if obs_flat.shape == mod_flat.shape:
        valid_mask = ~np.isnan(obs_flat) & ~np.isnan(mod_flat)
        obs_paired = obs_flat[valid_mask]
        mod_paired = mod_flat[valid_mask]
    else:
        obs_valid = obs_flat[~np.isnan(obs_flat)]
        mod_valid = mod_flat[~np.isnan(mod_flat)]
        min_len = min(len(obs_valid), len(mod_valid))
        obs_paired = obs_valid[:min_len]
        mod_paired = mod_valid[:min_len]

    if len(obs_paired) == 0: return np.nan, np.nan, np.nan, np.nan

    diff = mod_paired - obs_paired
    bias = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))

    std_obs = np.std(obs_paired)
    std_mod = np.std(mod_paired)
    sdr = float(std_mod / std_obs) if std_obs > 0 else np.nan

    min_val = min(np.min(obs_paired), np.min(mod_paired))
    max_val = max(np.max(obs_paired), np.max(mod_paired))
    if min_val == max_val:
        pss = 100.0 if np.allclose(obs_paired, mod_paired) else 0.0
    else:
        hist_obs, _ = np.histogram(obs_paired, bins=100, range=(min_val, max_val), density=True)
        hist_mod, _ = np.histogram(mod_paired, bins=100, range=(min_val, max_val), density=True)
        hist_obs = hist_obs / (np.sum(hist_obs) + 1e-8)
        hist_mod = hist_mod / (np.sum(hist_mod) + 1e-8)
        pss = float(np.sum(np.minimum(hist_obs, hist_mod)) * 100.0)

    return bias, mae, sdr, pss


# 4. LOOP PRINCIPAL
for cenario in CENARIOS:
    print(f"\n=======================================================")
    print(f"🚀 INICIANDO CENÁRIO FUTURO: {cenario.upper()}")
    print(f"=======================================================")

    caminho_csv = os.path.join(DIR_SAIDA_TABELAS, f"Metricas_Correcao_{cenario.upper()}.csv")

    if os.path.exists(caminho_csv):
        df_cenario = pd.read_csv(caminho_csv, sep=';', decimal=',')
        resultados_cenario = df_cenario.to_dict('records')
    else:
        df_cenario = pd.DataFrame()
        resultados_cenario = []

    for modelo in MODELOS:
        print(f"\n[GCM: {modelo}]")

        for var in VARIAVEIS:
            pasta_saida = os.path.join(BASE_SAIDA_NC, modelo, cenario, var)
            os.makedirs(pasta_saida, exist_ok=True)

            arquivos_esperados = [os.path.join(pasta_saida, f"{modelo}_{cenario}_{var}_corrigido_{ano}.nc4") for ano in
                                  ANOS_OVERLAP]
            if all(os.path.exists(arq) for arq in arquivos_esperados):
                metrica_pronta = False
                if not df_cenario.empty and ((df_cenario['GCM'] == modelo) & (df_cenario['Variável'] == var)).any():
                    metrica_pronta = True
                if metrica_pronta:
                    print(f" -> [{var}] ⏭️ Já processado e métricas salvas. A saltar...")
                    continue

            print(f" -> [{var}] A carregar matrizes brutas nativas...")
            ds_era5 = carregar_dados(os.path.join(BASE_ERA5, var), ano_inicio=1985, ano_fim=2025)
            ds_nex_hist = carregar_dados(os.path.join(BASE_NEX, modelo, 'historical', var), ano_inicio=1985,
                                         ano_fim=2014)
            ds_nex_fut = carregar_dados(os.path.join(BASE_NEX, modelo, cenario, var), ano_inicio=2015, ano_fim=2025)

            if ds_era5 is None or ds_nex_hist is None or ds_nex_fut is None:
                print(f" -> [{var}] ⚠️ Faltam dados brutos. A saltar.")
                if ds_era5: ds_era5.close()
                if ds_nex_hist: ds_nex_hist.close()
                if ds_nex_fut: ds_nex_fut.close()
                continue

            # --- A MÁGICA DA ADAPTAÇÃO DO ERA5 ---
            try:
                cal_modelo = ds_nex_hist.time.dt.calendar
            except AttributeError:
                cal_modelo = ds_nex_hist.indexes['time'].calendar

            if cal_modelo not in ['standard', 'gregorian', 'proleptic_gregorian']:
                print(f"    🔄 Convertendo ERA5 para o calendário do modelo: '{cal_modelo}'...")
                # Converte o calendário (remove 31s) e preenche os dias faltantes (29 e 30 Fev) com o clima do dia anterior
                ds_era5_compat = ds_era5.convert_calendar(cal_modelo, align_on='date').ffill(dim='time')
            else:
                ds_era5_compat = ds_era5
            # -------------------------------------

            print(f" -> [{var}] Alinhando grelhas e aplicando Método Delta...")
            ds_nex_hist = ds_nex_hist.reindex(lat=ds_era5_compat.lat, lon=ds_era5_compat.lon, method='nearest')
            ds_nex_fut = ds_nex_fut.reindex(lat=ds_era5_compat.lat, lon=ds_era5_compat.lon, method='nearest')

            # O Agrupamento Climatológico agora corre perfeito porque os calendários são gémeos!
            era5_hist_clim = ds_era5_compat.sel(time=PERIODO_HIST)[var].groupby('time.month').mean(dim='time')
            nex_hist_clim = ds_nex_hist.sel(time=PERIODO_HIST)[var].groupby('time.month').mean(dim='time')
            ds_fut_overlap = ds_nex_fut.sel(time=PERIODO_OVERLAP).copy()

            if var in ['tasmax', 'tasmin']:
                delta = era5_hist_clim - nex_hist_clim
                ds_fut_overlap[var] = ds_fut_overlap[var].groupby('time.month') + delta

            elif var in ['pr', 'rsds', 'sfcWind']:
                delta = (era5_hist_clim + 0.1) / (nex_hist_clim + 0.1)
                delta = delta.clip(max=5.0)
                ds_fut_overlap[var] = ds_fut_overlap[var].groupby('time.month') * delta
                ds_fut_overlap[var] = ds_fut_overlap[var].clip(min=0)

            elif var == 'hurs':
                delta = era5_hist_clim - nex_hist_clim
                ds_fut_overlap[var] = ds_fut_overlap[var].groupby('time.month') + delta
                ds_fut_overlap[var] = ds_fut_overlap[var].clip(min=0, max=100)

            encoding = {var: {'zlib': True, 'complevel': 5, '_FillValue': np.nan, 'dtype': 'float32'}}
            for ano, arquivo_ano_saida in zip(ANOS_OVERLAP, arquivos_esperados):
                if not os.path.exists(arquivo_ano_saida):
                    ds_ano = ds_fut_overlap.sel(time=str(ano))
                    ds_ano.to_netcdf(arquivo_ano_saida, engine='h5netcdf', encoding=encoding)

            print(f" -> [{var}] Extraindo métricas avançadas (Bias, MAE, SDR, PSS)...")

            era5_overlap_clim = ds_era5_compat.sel(time=PERIODO_OVERLAP)[var].groupby('time.month').mean(dim='time')
            nex_fut_overlap_bruto_clim = ds_nex_fut.sel(time=PERIODO_OVERLAP)[var].groupby('time.month').mean(
                dim='time')
            nex_fut_overlap_corr_clim = ds_fut_overlap[var].groupby('time.month').mean(dim='time')

            era5_overlap_daily = ds_era5_compat.sel(time=PERIODO_OVERLAP)[var]
            nex_fut_bruto_daily = ds_nex_fut.sel(time=PERIODO_OVERLAP)[var]
            nex_fut_corr_daily = ds_fut_overlap[var]

            bias_b, mae_b, sdr_b, _ = calcular_metricas_avancadas(era5_overlap_clim, nex_fut_overlap_bruto_clim)
            bias_c, mae_c, sdr_c, _ = calcular_metricas_avancadas(era5_overlap_clim, nex_fut_overlap_corr_clim)

            _, _, _, pss_b = calcular_metricas_avancadas(era5_overlap_daily, nex_fut_bruto_daily)
            _, _, _, pss_c = calcular_metricas_avancadas(era5_overlap_daily, nex_fut_corr_daily)

            resultados_cenario.append({
                "Variável": var,
                "GCM": modelo,
                "Cenário": cenario.upper(),
                "BIAS Bruto": bias_b, "MAE Bruto": mae_b, "SDR Bruto": sdr_b, "PSS Bruto": pss_b,
                "BIAS Corr": bias_c, "MAE Corr": mae_c, "SDR Corr": sdr_c, "PSS Corr": pss_c
            })

            df_cenario = pd.DataFrame(resultados_cenario)
            df_cenario.to_csv(caminho_csv, index=False, sep=';', decimal=',', float_format='%.4f')

            ds_era5.close()
            ds_nex_hist.close()
            ds_nex_fut.close()
            del ds_fut_overlap
            gc.collect()

print("\n🎉 MEGA PROCESSAMENTO LOCAL CONCLUÍDO! AS TABELAS FINAIS ESTÃO PRONTAS.")