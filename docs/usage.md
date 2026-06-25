# Usage

Install dependencies from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Launch the viewer:

```powershell
streamlit run app/dashboard.py
```

Upload:

- a CT image volume in `.nii` or `.nii.gz` format
- a matching mask in `.nii` or `.nii.gz` format

The image and mask must be 3D volumes with the same shape.

In the slice viewer, use the display mode selector to switch between:

- Raw CT slice
- Liver windowed CT slice

The liver-windowed view uses a Hounsfield window with center `60` and width `150`. The mask and overlay remain available in both display modes. The metrics panel also shows basic HU statistics for the uploaded image volume.
