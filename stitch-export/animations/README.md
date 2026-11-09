# Particle Typography Effect – Animation Specifications & Integration Guide

This directory contains the standalone implementation of the high-fidelity **Particle Typography Effect** inspired by `st.dev`, tailored for the **AIoT Plant Health Monitoring System**.

---

## 1. Visual & Architectural Overview

- **Core Concept**: An ethereal cloud of microscopic luminous particles forming, dissolving, and re-coalescing around the left side of the project title.
- **Typography**: Clean, geometric sans-serif using **Space Grotesk** (`wght: 700`), with an emerald gradient accent on the `AIoT` prefix.
- **Particle Dynamics**:
  - Sampled raster coordinates from the text glyphs produce genuine edge disintegration.
  - Smooth left-to-right drift with gentle sinusoidal vertical perturbation.
  - Particles partially overlap the beginning of the title and disperse downstream.
- **Palette**:
  - **Base Particles**: Pure white (`#FFFFFF`) with semi-transparent halos.
  - **Health / Normal Embers**: Emerald green (`#4EDEA3` / `#10B981`) telemetry highlights.
  - **Blight / Anomaly Embers**: Crimson red (`#FF5252` / `#EF4444`) subtle warning micro-accents.
  - **Telemetry Ambient**: Electric cyan (`#4CD7F6`) secondary accents.
- **Rendering Architecture**: High-DPI canvas (`window.devicePixelRatio`) with `globalCompositeOperation = 'screen'` and variable soft glow (`shadowBlur`).

---

## 2. File Manifest

| File | Purpose |
| :--- | :--- |
| `particle_typography.html` | Self-contained, responsive HTML5 Canvas component with interactive tuning controls, live telemetry badges, and 60 FPS loop. |
| `README.md` | Integration and parameter documentation. |

---

## 3. How to Integrate into Streamlit

When ready to incorporate this animation into your Streamlit dashboard header without modifying your core backend:

```python
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Path to the particle typography HTML file
ANIMATION_PATH = Path("stitch-export/animations/particle_typography.html")

def render_particle_header():
    if ANIMATION_PATH.exists():
        with open(ANIMATION_PATH, "r", encoding="utf-8") as f:
            html_code = f.read()
        # Render inside Streamlit header container
        components.html(html_code, height=170, scrolling=False)
    else:
        st.title("AIoT Plant Health Monitoring System")
```

---

## 4. Key Configurable Parameters

In `particle_typography.html`, the following parameters can be adjusted directly in the script:

- `densitySlider` / `targetCount`: Default `280` particles. Controls cloud volume.
- `flowSpeed`: Default `1.8`. Controls horizontal drift velocity.
- `glowStrength`: Default `8`. Controls soft glow blur radius.
- `COLORS`: Color definition object for custom palette matching.
- `mouse.radius`: Default `100px`. Interactive mouse repulsion field.

