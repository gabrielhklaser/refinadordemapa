# RefinaPaleo — Plataforma de Refinamento de Dados Paleoclimáticos

Plataforma web que processa **mapas paleoclimáticos em PDF vetorizado**, corrige
descontinuidades visuais entre zonas climáticas **sem alterar as geometrias
geográficas originais** e entrega os resultados em **TIFF/PNG** com conferência
de confiabilidade — implementando a skill `refinamento-mapas-paleo`.

## Fluxo (skill)

1. **Carregamento e desacoplamento** — o PDF é rasterizado e cada primitiva
   vetorial é classificada: zonas climáticas (elegíveis ao refinamento),
   camadas geográficas protegidas (costa, graticule, molduras), texto e
   pontos observacionais. A área climática é convertida para CIELAB.
2. **Suavização morfológica / denoising** — 3 tecnologias distintas.
3. **Realce com rigor científico** — refinamento confinado a uma faixa
   estreita nas fronteiras das zonas; camadas protegidas restauradas *bit a bit*.
4. **3 versões por tecnologia** — Mínima, Intermediária e Intensa.
5. **Conferência de confiabilidade** — % de mudança das zonas (ΔE CIE76 > 3),
   erro vetorial/observacional (0 px por construção), correlação cruzada das
   camadas protegidas, PSNR/SSIM.
6. **Exportação TIFF/PNG** — com metadados científicos embutidos no TIFF.

## Tecnologias de refinamento

| # | Tecnologia | Parâmetros (Mínima → Intensa) |
|---|------------|-------------------------------|
| MORF-MS | Suavização Morfológica Multiescala (fechamento→abertura, kernel elíptico, Lab) | r = 3 → 9 px |
| BILAT-EP | Filtragem Bilateral Edge-Preserving iterativa | σcor 18 → 48 |
| DIFF-PM | Difusão Anisotrópica de Perona–Malik (Lab) | κ 10 → 24, 12 → 44 iterações |

### Tipos de PDF suportados

- **Zonas vetoriais** — preenchimentos vetoriais por zona climática.
- **Híbrido (campo rasterizado embutido)** — mapas interpolados (ex.: kNN+IDW
  com gradiente) exportados com o campo climático como imagem dentro do PDF e
  o restante (costa, graticule, pontos, textos) em vetor. A imagem é detectada
  automaticamente e tratada como região climática elegível ao refinamento.

Dois mapas de exemplo acompanham a plataforma: zonas vetoriais (LGM, 21 ka) e
aridez interpolada (115 Ma, estilo `knn_idw_gradient` com campo rasterizado).

## Seção Artística (modo pictórico)

Além do fluxo científico, a plataforma oferece a **Seção Artística**: rigor
científico suspenso, o mapa é reinterpretado em 4 estilos pictóricos × 3
intensidades (12 obras), mantendo **apenas** duas preservações absolutas:

- **Pontos observacionais e vetores geográficos não se movem** (restaurados bit a bit);
- **As zonas climáticas não mudam de lugar** (núcleo das fronteiras com os pixels originais;
  o oceano/fundo permanece original, fora da pintura).

Estilos: 🖌️ Aquarela · 🎨 Pintura a Óleo · 🌈 Cartoon/Pôster · 🌸 Pastel Sonhador.
Pontos perdidos (*specks*) são absorvidos por inpainting antes da pintura.

## Como rodar

```bash
pip install -r requirements.txt
python3 -m uvicorn --app-dir app main:app --host 0.0.0.0 --port 8000
```

Abra `http://localhost:8000/`, envie um PDF vetorizado (ou clique em
**Usar mapa de exemplo**, que gera um mapa paleoclimático ilustrativo da
América do Sul no Último Máximo Glacial, 100% vetorial) e acompanhe o
processamento. Os resultados, a tabela de confiabilidade e os downloads
TIFF/PNG ficam em `/resultados?job=<id>`.

## Estrutura

```
app/
  main.py         API FastAPI (upload, jobs, downloads, zip, modo sci/art)
  pipeline.py     desacoplamento de camadas, 4 tecnologias, confiabilidade, exportação
  artistic.py     Seção Artística (4 estilos pictóricos × 3 intensidades)
  sample_map.py   gerador dos mapas de exemplo (zonas vetoriais + aridez rasterizada)
static/           páginas (upload + resultados), CSS e JS
data/jobs/        artefatos gerados por job (gitignored)
```
