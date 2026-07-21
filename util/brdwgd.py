"""
Loader para o BR-DWGD (Brazilian Daily Weather Gridded Data, Xavier et al. 2022).

Le os arquivos NetCDF locais (um arquivo por variavel, ja cobrindo toda a
serie historica) e extrai series temporais por ponto (lat/lon), convertendo
para o formato de entrada esperado pelo WOFOST (mesmo padrao usado por
SensitivityAnalyzer.NetCDFDataLoader.convert_to_wofost_format para o pipeline
de milho, baseado em NASA POWER).

Fonte dos dados: https://github.com/AlexandreCandidoXavier/BR-DWGD
"""
import os

import numpy as np
import pandas as pd
import xarray as xr
from pcse.util import vap_from_relhum

# Token do arquivo -> nome da variavel dentro do NetCDF.
# Ex.: Tmax_20010101_20251232_BR_DWGD_UFES_UTEXAS_v_3.2.4.nc
BRDWGD_VARIABLES = {
    'Tmax': 'Tmax',  # graus C
    'Tmin': 'Tmin',  # graus C
    'RH': 'RH',      # umidade relativa, %
    'u2': 'u2',      # velocidade do vento a 2 m, m/s
    'Rs': 'Rs',       # radiacao solar, MJ/m2/dia
    'pr': 'pr',      # precipitacao, mm/dia
}

DEFAULT_FILENAME_TEMPLATE = "{var}_20010101_20251231_BR-DWGD_UFES_UTEXAS_v_3.2.4.nc"


class BRDWGDLoader:
    """
    Abre as 6 variaveis do BR-DWGD uma unica vez (lazy, via dask) e permite
    extrair a serie temporal de qualquer ponto (lat, lon) por vizinho mais
    proximo da grade.
    """

    def __init__(self, data_dir, filename_template=DEFAULT_FILENAME_TEMPLATE,
                 lat_name='latitude', lon_name='longitude', time_name='time'):
        self.data_dir = data_dir
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.time_name = time_name
        self._datasets = {}

        for token in BRDWGD_VARIABLES:
            fpath = os.path.join(data_dir, filename_template.format(var=token))
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Arquivo do BR-DWGD nao encontrado para a variavel '{token}': {fpath}"
                )
            self._datasets[token] = xr.open_dataset(fpath, chunks={self.time_name: 'auto'})

    def close(self):
        for ds in self._datasets.values():
            ds.close()
        self._datasets = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _data_array(self, token):
        ds = self._datasets[token]
        var_name = BRDWGD_VARIABLES[token]
        if var_name in ds.data_vars:
            return ds[var_name]
        # Fallback: usa a unica variavel de dados presente no arquivo, caso o
        # nome interno nao bata exatamente com o token do arquivo.
        data_vars = list(ds.data_vars)
        if len(data_vars) == 1:
            return ds[data_vars[0]]
        raise KeyError(
            f"Nao foi possivel identificar a variavel '{token}' no dataset "
            f"({self._datasets[token].encoding.get('source', '?')}). "
            f"Variaveis disponiveis: {data_vars}"
        )

    def extract_point_series(self, lat, lon, start_date=None, end_date=None):
        """
        Extrai a serie diaria das 6 variaveis para o ponto da grade mais
        proximo de (lat, lon).

        Args:
            lat, lon: coordenadas do ponto (graus decimais)
            start_date, end_date: limites opcionais (str 'YYYY-MM-DD' ou datetime)

        Returns:
            pd.DataFrame com colunas date, Tmax, Tmin, RH, u2, Rs, pr
        """
        series = {}
        for token in BRDWGD_VARIABLES:
            da = self._data_array(token)
            point = da.sel({self.lat_name: lat, self.lon_name: lon}, method='nearest')
            if start_date is not None or end_date is not None:
                point = point.sel({self.time_name: slice(start_date, end_date)})
            series[token] = point.to_series()

        df = pd.DataFrame(series)
        df.index.name = 'date'
        df = df.reset_index()
        df.rename(columns={self.time_name: 'date'}, inplace=True)
        return df

    @staticmethod
    def to_wofost_frame(df):
        """
        Converte o DataFrame bruto do BR-DWGD (Tmax, Tmin, RH, u2, Rs, pr) para
        o conjunto de variaveis/unidades que o WOFOST espera:
        IRRAD [J/m2/d], TMIN/TMAX [degC], VAP [hPa], RAIN [cm/d], WIND [m/s].
        """
        tmin = df['Tmin'].to_numpy(dtype=float)
        tmax = df['Tmax'].to_numpy(dtype=float)
        tmean = (tmin + tmax) / 2.0

        irrad = df['Rs'].to_numpy(dtype=float) * 1_000_000.0  # MJ/m2/d -> J/m2/d
        rain = df['pr'].to_numpy(dtype=float) / 10.0            # mm/d -> cm/d
        wind = df['u2'].to_numpy(dtype=float)                   # ja e vento a 2 m
        rh = df['RH'].to_numpy(dtype=float)

        vap = np.array([
            vap_from_relhum(rh_val, t_val) * 10.0  # kPa -> hPa
            if not (pd.isna(rh_val) or pd.isna(t_val)) else np.nan
            for rh_val, t_val in zip(rh, tmean)
        ])

        return pd.DataFrame({
            'date': pd.to_datetime(df['date'].values),
            'IRRAD': irrad,
            'TMIN': tmin,
            'TMAX': tmax,
            'T2M': tmean,
            'VAP': vap,
            'RAIN': rain,
            'WIND': wind,
        })
