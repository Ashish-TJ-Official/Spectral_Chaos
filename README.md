# 🔬 Spectral Analysis Web Application

An interactive, high-performance web application built with Python and Flask for analyzing, identifying, and unmixing geological and material spectral data. Powered by dynamic visualization tools and numerical computing algorithms, it offers a seamless, app-like experience for working with complex spectral libraries.

---

## 🌟 Key Features

- **Interactive Library Explorer:** Dynamically navigate complex hierarchical datasets (`Type` → `Class` → `Subclass` → `Texture`) parsed from custom JSON schemas and stream Base64-encoded Matplotlib plots directly to the UI canvas without page reloads.
- **Automated Mineral Identification:** Drop in raw spectral datasets (`.csv` or `.txt` wavelength/reflectance grids) to identify unknown target samples using multi-metric similarity algorithms:
  - **SAM** (Spectral Angle Mapper)
  - **SID** (Spectral Information Divergence)
  - **SFF** (Spectral Feature Fitting)
  - Interactive spectral overlays showing target vs. best-matching library spectra.
- **Linear Unmixing (NNLS):** Build custom multi-endmember collections and calculate exact percentage breakdowns for composite mixture inputs using Non-Negative Least Squares (NNLS) alongside Reconstruction Error ($R_{NORM}$) indicators.
- **Seamless Single-Page Experience:** Fully asynchronous frontend powered by JavaScript `fetch` calls to backend `/api/` routes, delivering dynamic DOM updates with zero full-page refreshes.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.x, Flask
- **Data & Numerical Analytics:** NumPy, SciPy, Pandas, PyArrow (Parquet engine)
- **Visualization:** Matplotlib (Base64 renderer)
- **Algorithms:** Non-Negative Least Squares (NNLS), SAM, SID, SFF
- **Frontend:** HTML5, CSS3, JavaScript (Fetch API, dynamic DOM Manipulation)

---

## 📂 Project Architecture

```text
├── app.py                            # Core Flask web server & API routes
├── requirements.txt                  # Python dependencies
├── Spectral_Library_Hierarchy.json   # Hierarchical spectral metadata schema
├── Spectral_Library.parquet          # High-performance binary spectral database
└── templates/                        # Jinja2 HTML templates
    ├── index.html                    # Library Explorer interface
    ├── mineral-identification.html   # Identification module interface
    └── linear-unmixing.html          # Linear Unmixing module interface
