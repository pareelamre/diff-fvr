# Diff-FVR Manuscript Sources (`paper/`)

This directory tracks the LaTeX manuscript sources and submission text for **Diff-FVR / PoU-FVR**:

| File | Target Venue | Notes |
| :--- | :--- | :--- |
| `main_ieee_tmi.tex` | IEEE Transactions on Medical Imaging (TMI) | Uses `ieeecolor` + `tmi.sty` + `LOGO-tmi-web.png` |
| `main_media_elsarticle.tex` | Medical Image Analysis (MedIA) | Uses `elsarticle` (`highlights_media.txt` included) |
| `main_miccai_llncs.tex` | MICCAI (double-blind review) | Uses `llncs` (anonymized) |
| `main_arxiv.tex` | arXiv preprint | Self-contained standard `IEEEtran` (compiles without TMI templates) |
| `cover_letter.md` | Editorial cover letter | Summary of verified results through `exp33` (`v1.2.0`) |

## Building PDFs Locally

All `.tex` files expect figures under `figures/fig*.png`, which are generated into `../outputs/paper_figures/` by `../experiments/make_paper_figures_js.py`. To compile `main_arxiv.tex` cleanly in a temporary directory:

```powershell
mkdir build_arxiv
cp paper/main_arxiv.tex build_arxiv/main.tex
cp -r outputs/paper_figures build_arxiv/figures
tectonic build_arxiv/main.tex
```
