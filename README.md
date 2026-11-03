# dauer-nociception
Dauer nociception circuit analysis code for Choe et al.

## Repository structure

| Folder | Contents |
|--------|----------|
| `basic_analysis/` | Behavior analysis and EM mesh/skeleton analysis |
| `calcium_imaging/` | Calcium imaging analysis |
| `avidance_initiation/` | Avoidance initiation circuit simulations in Fig. 6 |
| `activity_modeling/` | AVB activity modeling in Fig. 3 (see [activity_modeling/README.md](activity_modeling/README.md)) |

All analyses are Jupyter notebooks.

## Installation

**Python 3.10–3.12** is supported; the code was tested with **Python 3.12**. Python 3.13+ is not recommended because some dependencies (`cloud-volume`, `vtk`) may not yet provide prebuilt packages for it.

```bash
git clone https://github.com/conncslab/dauer-nociception.git
cd dauer-nociception

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Register the environment as a Jupyter kernel (optional)

This lets you select the environment from Jupyter or VS Code:

```bash
python -m ipykernel install --user --name nociception --display-name "Python (nociception)"
```

### Verify the installation

```bash
python -c "import numpy, scipy, pandas, openpyxl, matplotlib, seaborn, sklearn, networkx, tqdm, tifffile, cv2, PIL, mrcfile, nd2, cloudvolume, vtk; print('OK')"
```

## Running the notebooks

Start Jupyter from the repository root and open a notebook:

```bash
jupyter lab
```

Notebooks load data with paths relative to their own folder (e.g. `./data/...`), so run each notebook with its own folder as the working directory. Jupyter Lab and VS Code do this by default.

To run a notebook non-interactively from the command line:

```bash
cd activity_modeling
jupyter nbconvert --to notebook --execute --inplace activity_modeling.ipynb
```
