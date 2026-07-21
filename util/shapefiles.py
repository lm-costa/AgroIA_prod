import os
import geopandas as gpd
import geodatasets

print("1. Carregando o mapa físico global...")
# Puxa o mapa físico apenas com as massas de terra
caminho = geodatasets.get_path('naturalearth.land')
gdf_mundo = gpd.read_file(caminho)

print("2. Removendo a Antártida...")
# Usa o indexador espacial (.cx) para manter apenas o que está acima da latitude -60°
gdf_sem_antartida = gdf_mundo.cx[:, -60:]

print("3. Processando a geometria (isso pode levar alguns segundos)...")
# Une os polígonos restantes para otimizar o futuro recorte
gdf_continentes = gdf_sem_antartida.dissolve()

# 4. Salvando no Drive
CAMINHO_SALVAR = r'D:\OneDrive\Documentos\Git\Doutorado\inputs\data\shape'

os.makedirs(os.path.dirname(CAMINHO_SALVAR), exist_ok=True)
print(f"4. Salvando o novo shapefile em: {CAMINHO_SALVAR} ...")

# Exporta e sobrescreve o arquivo antigo
gdf_continentes.to_file(CAMINHO_SALVAR)

print("✅ Shapefile atualizado com sucesso no Drive! Agora ele não possui mais a Antártida.")