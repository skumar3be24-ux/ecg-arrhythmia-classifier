<div align="center">

# 🫀 ECG Arrhythmia Classifier on a Zynq FPGA

### Real-time heartbeat classification on the PYNQ-Z2 — Pan-Tompkins QRS detection + INT8 1D-CNN, sharing a single DSP

![Board](https://img.shields.io/badge/Board-PYNQ--Z2-blue?style=for-the-badge)
![FPGA](https://img.shields.io/badge/FPGA-Zynq%20XC7Z020-red?style=for-the-badge)
![HDL](https://img.shields.io/badge/HDL-Verilog-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-INT8%20CNN-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**97.05 %** on-board accuracy &nbsp;•&nbsp; **49,027 / 49,027** beats bit-exact &nbsp;•&nbsp; **12.2 ms** per beat &nbsp;•&nbsp; **2 → 1** shared DSP

</div>

---

## 📌 Overview

A complete ECG beat classifier running on a single Xilinx Zynq-7020 chip.

- 🔍 A **Pan-Tompkins QRS detector** finds every heartbeat in the programmable logic (PL).
- 🧠 An **INT8 1D convolutional neural network** (8,101 parameters) classifies each beat in the PL.
- ⚙️ The **ARM processor** streams the ECG, corrects the detected R-peaks and computes the RR-interval features.
- ♻️ The detector's squaring multiplier and the CNN's requantisation multiplier **share one DSP48E1** (measured 2 → 1 DSP after place-and-route).

Every beat is labelled with one of the five AAMI EC57 classes: **N**ormal, **S**upraventricular, **V**entricular, **F**usion, **Q**/unknown.

<p align="center">
  <img src="paper_figures/png/fig01_system_architecture.png" width="900" alt="System architecture on the PYNQ-Z2">
</p>

---

## 🏆 Key results

<div align="center">

| 🎯 Patient-adaptive accuracy | 🧪 Inter-patient accuracy | ✅ Bit-exact on board | ⚡ CNN latency |
|:---:|:---:|:---:|:---:|
| **97.05 %** | **94.60 %** | **49,027 / 49,027** | **12.2 ms / beat** |
| measured on the board | unseen patients, INT8 model | 22 / 22 records | at 50 MHz |

</div>

| Metric | Value | Source |
|---|---|---|
| Accuracy, patient-adaptive (DS2, min 5-30, 40,816 beats) | **97.05 %** (strict 95.48 %, macro-F1 0.821) | Measured on the board, 9 Oct 2026 |
| Accuracy, inter-patient (DS1 → DS2, 48,794 beats) | **94.60 %** (strict 92.95 %, macro-F1 0.619) | Bit-exact INT8 software model |
| Board vs reference model | 49,027 / 49,027 beats, 22 / 22 records | Measured on the board |
| CNN latency | 12.196 ms (609,816 cycles); 12.68 ms mean on board incl. polling | RTL simulation / board |
| Clock & timing | 50 MHz, WNS +5.478 ns, WHS +0.024 ns | Vivado post-route |
| Resources | 6,146 LUT (11.6 %) · 4,560 FF (4.3 %) · 4.5 BRAM (3.2 %) · 2 DSP (0.9 %) | Vivado post-route |
| Power | 1.418 W on chip (accelerator 25 mW) | Vivado estimate, not measured |

> [!NOTE]
> **97.05 %** uses the AAMI *patient-adaptive* protocol: the first 5 minutes of each test record are used for training (one global model) and only minutes 5-30 are scored. On patients **never seen** in training, the same hardware reaches **94.60 %**.

<p align="center">
  <img src="paper_figures/png/fig03_accuracy_per_record_onboard.png" width="900" alt="On-board accuracy per test record">
</p>

---

## ✨ What makes it different

| | |
|---|---|
| 🔗 **Detector in the loop** | Beats come from the on-chip QRS detector, not from cardiologist annotations, so detector errors are counted in the accuracy. |
| ♻️ **One shared multiplier** | The QRS squarer and the CNN requantiser time-share one DSP48E1 — measured 2 → 1 DSP. |
| 🎯 **Bit-exact hardware** | Every beat on the board matches the Python INT8 reference model exactly. |
| 📏 **Honest protocol** | No synthetic oversampling (SMOTE), no random patient mixing; both patient-adaptive and inter-patient results are reported. |
| 🪶 **Tiny footprint** | Whole system uses ~12 % of LUTs on a low-cost Zynq-7020. |

---

## 📂 Repository layout

```
ecg-arrhythmia-classifier/
├── hardware/        Verilog RTL, CNN weights, self-checking testbenches
│   ├── rtl/         top/ (AXI + shared multiplier), qrs/ (Pan-Tompkins), cnn/ (INT8 CNN)
│   ├── weights/     weight/bias memory files, quantisation params, golden vectors
│   └── sim/         testbenches, test vectors, run_all.sh / run_all.ps1
├── software/        Python: dataset, model, training, quantisation, evaluation, PYNQ runtime
├── fpga/vivado/     Tcl scripts to rebuild the Vivado project and bitstream
├── pynq/            ready bitstream + hwh, board test vectors, run instructions
├── results/         board run, evaluations, models, Vivado reports, design-space study
├── paper_figures/   all result figures (PNG + PDF) and the script that draws them
├── docs/            project guide, final report, development log, accuracy analysis
└── data/            MIT-BIH download location (not tracked)
```

📖 Every folder has its own `README.md` explaining its files.

---

## 🚀 Quick start

### 1️⃣ Run on the PYNQ-Z2 board

```bash
python pynq/make_pynq_package.py        # builds pynq/ecg_accel_pynq_package.zip
```
Upload the zip to the board's Jupyter, unzip it, then:
```python
import os, sys; os.chdir('ecg_accel/software'); sys.path.insert(0, '.')
from pynq import Overlay; Overlay('../pynq/overlay/ecg_accel.bit')
import board_test as T; T.run_all(realtime=False)     # 22 records, ~50 min
import board_summary as S; S.summary()                # expect ALL PASS and 97.05 %
```

### 2️⃣ Simulate the hardware (Icarus Verilog 11+)

```bash
cd hardware/sim && ./run_all.sh          # Windows: .\run_all.ps1
```

### 3️⃣ Reproduce the software results (Python 3.10+)

```bash
pip install -r requirements.txt
cd software
python mitbih.py --download              # MIT-BIH into data/mitdb (~100 MB)
python beat_dataset.py                   # beat datasets
python train.py                          # train the CNN
python quantize.py                       # INT8 weights -> hardware/weights/
python evaluate.py ../hardware/weights PA_test                # 97.05 %
python evaluate.py ../results/models/inter_patient_int8 DS2   # 94.60 %
python make_board_vectors.py             # board test vectors
python tests/mock_board_test.py          # PS runtime vs software model of the PL
```

### 4️⃣ Rebuild the bitstream (Vivado 2025.1)

```bash
cd fpga/vivado
vivado -mode batch -source create_project.tcl -notrace
vivado -mode batch -source build.tcl -notrace
```

### 5️⃣ Regenerate all figures

```bash
python paper_figures/make_figures.py
```

---

## ✅ Verification

| Check | Result |
|---|---|
| All six RTL testbenches (`hardware/sim/run_all.sh`) | ✅ Pass — system 64/64 classes & RR words; CNN 64/64 at three contention levels; filters, peak detector, RR engine, shared multiplier bit-exact |
| `train.py` re-run (CPU, seed 0) | ✅ Same epoch (16), identical weights |
| `beat_dataset.py` + `quantize.py` from raw MIT-BIH | ✅ Byte-identical to `hardware/weights/` |
| `evaluate.py hardware/weights PA_test` | ✅ 97.05 %, identical confusion matrix |
| `make_board_vectors.py` | ✅ Byte-identical to the board vectors |
| `mock_board_test.py` (records 113, 202, 228) | ✅ Bit-exact, same accuracy as the board |
| Board run, 9 Oct 2026 | ✅ 22 / 22 records PASS, 97.05 % |

The bitstream in `pynq/overlay/` is the one used for the board run. It was built from the same HDL before files were renamed for release ([`docs/NAMING_MAP.md`](docs/NAMING_MAP.md)); the renamed sources pass every testbench.

---

## ⚠️ Limitations

- ≥ 97 % holds only under the patient-adaptive protocol; inter-patient accuracy is 94.60 %.
- S and F classes are much weaker than N and V (F sensitivity 1.8 % inter-patient).
- Power is a Vivado estimate, not a bench measurement.
- The board replays recorded MIT-BIH signals; there is no analogue ECG front end.
- 360 → 180 Hz decimation and peak post-processing run on the ARM, not in the FPGA.

---

## 📚 Documentation

| Document | What it covers |
|---|---|
| [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md) | Full explanation, from background to file-by-file reference |
| [`docs/report/ECG_FPGA_Project_Report.pdf`](docs/report/ECG_FPGA_Project_Report.pdf) | Final project report |
| [`docs/development_log.md`](docs/development_log.md) | Dated record of every measurement and bug |
| [`docs/NAMING_MAP.md`](docs/NAMING_MAP.md) | Development names → release names |
| [`paper_figures/FIGURE_CAPTIONS.md`](paper_figures/FIGURE_CAPTIONS.md) | Captions and data sources for all figures |

---

## 🙏 Data and acknowledgements

- **ECG data:** MIT-BIH Arrhythmia Database (Moody & Mark, *IEEE Eng. Med. Biol.* 20(3):45-50, 2001), via PhysioNet (Goldberger et al., *Circulation* 101(23):e215-e220, 2000), ODC Attribution License v1.0. Raw records are not stored here; `software/mitbih.py --download` fetches them.
- **QRS detection:** Pan & Tompkins, *IEEE Trans. Biomed. Eng.* 32(3):230-236, 1985.
- **Inter-patient split:** de Chazal et al., *IEEE Trans. Biomed. Eng.* 51(7):1196-1206, 2004.

---

## 📄 Citation and license

If you use this work, please cite it using [`CITATION.cff`](CITATION.cff).
Code and HDL are released under the [MIT License](LICENSE).

<div align="center">

⭐ **If you found this useful, consider giving the repo a star!** ⭐

</div>
