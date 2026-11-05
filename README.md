# dauer-nociception
Dauer nociception circuit analysis code for Choe et al.

## Repository structure

| Folder | Contents |
|--------|----------|
| `basic_analysis/` | Behavior analysis and EM mesh/skeleton analysis |
| `calcium_imaging/` | Calcium imaging analysis |
| `avidance_initiation/` | Avoidance initiation circuit simulations in Fig. 6 |
| `activity_modeling/` | AVB activity modeling in Fig. 3 ([AVB_activity_modeling.ipynb](activity_modeling/AVB_activity_modeling.ipynb)) |

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
