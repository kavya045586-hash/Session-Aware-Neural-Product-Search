# 🔍 Session-Aware Neural Product Search
### *An Expert Shopping Assistant that understands WHO YOU ARE based on what you click*

> **Standard Search**: You search "camera" → shows more cameras.  
> **This Project**: You like a camera, a lens, and a battery → the system understands you are a **Professional Photographer** and discovers things you didn't even search for, like high-speed memory cards or professional power backups.

---

## 🧠 How It Works (The 3-Step Process)

### Step 1: The Memory (GRU)
Most systems forget what you clicked 2 minutes ago. This system uses a **GRU (Gated Recurrent Unit)** to remember the **order** of your likes.
- *Example*: Like 1 (Mic) + Like 2 (Ring Light) = The system "remembers" you are building a **studio**.

### Step 2: The Map (Two-Tower Model)
The system places all **1,070,699 products** on a giant mathematical "map". Products that are usually bought together (like a camera and a tripod) are placed very close to each other on this map.

### Step 3: The Super-Fast Search (FAISS)
Because there are over **1 million items**, searching one by one would be too slow. The system uses **FAISS**, a special tool that scans the entire map and finds the best matches in a tiny fraction of a second.

---

## 🏗️ What Was Built

| Component | Description |
|---|---|
| **The Brain** | A Neural Network trained on 200,000 real customer sessions |
| **The Library** | A database of over 1 million electronics products |
| **The Interface** | A website (Flask) where you can search, "Like" items with a heart, and see the AI update predictions instantly |

---

## 📁 Project Structure

```
Session-Aware-Neural-Product-Search/
│
├── 01_data_merge.py        # Parse raw Amazon data, merge reviews + metadata
├── 02_prepare_search.py    # Build Title → ASIN search dictionary
├── 03_preprocess.py        # Create sliding window user sequences for training
├── 04_train.py             # Train the Two-Tower GRU model
├── 05_build_faiss.py       # Extract item embeddings, build FAISS index
├── 06_evaluate.py          # Evaluate model metrics
├── 06_search_map.py        # Build Title → Index mapping for the app
├── app.py                  # Flask web application
├── model_arch.py           # Neural network architecture
├── templates/
│   └── index.html          # Neural Discovery Engine frontend
└── requirements.txt
```

---

## 📊 Real Evaluation Results

```
==========================================
✨ MAJOR PROJECT EVALUATION REPORT ✨
==========================================
📈 Hit Rate@10:      0.3600%   (Accuracy)
📉 MRR@10:           0.1597%   (Ranking Quality)
🌐 Catalog Coverage: 0.1642%   (Model Diversity)
👤 Personalization:  100.0000% (User Uniqueness)
==========================================
```

> **Personalization score of 100%** means every single user gets a completely unique recommendation list — no two users see the same results.

---

## 🎬 Live Demo — Content Creator Session

This is a real example of the system understanding user intent step by step:

**Step 1 — Like: Blue Yeti Microphone (Blackout)**
- Model generates initial vector for high-quality audio recording
- Recommends: Western Digital 4TB Hard Drive + SanDisk Flash Drive
- *Why?* People recording high-quality audio need massive storage space

**Step 2 — Like: 10 Inch Selfie Ring Light with Tripod**
- GRU combines "Audio" + "Lighting" signals
- Recommendations shift to: Panasonic ErgoFit Earbuds + Bluetooth Headphones
- *Why?* The model now realizes this isn't just audio — this is a **Video Creator** who needs monitoring gear

**Step 3 — Like: Webcam Tripod for Logitech C920**
- The "Aha!" moment. Model now sees sequence: **Pro Mic → Lighting → Webcam Gear**
- Recommends: CyberPower UPS System + Kingston HyperX 16GB RAM
- *Why?* A streamer or heavy video editor needs a UPS so their computer doesn't crash during a render/stream, and high-performance RAM to handle editing software

### 🎯 Exam Demo Script
*"Observe how the system starts by suggesting storage for a single mic, but as I add lighting and webcam gear, it correctly predicts that a Content Creator needs high-performance RAM and a UPS for their workstation."*

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/kavya045586-hash/Session-Aware-Neural-Product-Search.git
cd Session-Aware-Neural-Product-Search
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Setup

Dataset files are not included due to size. Download from [Amazon Review Data (UCSD)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) and place in the `data/` folder:

```
data/
├── Electronics.jsonl.gz
└── meta_Electronics.jsonl.gz
```

---

## 🚀 Run Order (First Time)

```bash
python 01_data_merge.py       # Merge raw Amazon data (~20-40 min)
python 02_prepare_search.py   # Build keyword search map (~1 min)
python 03_preprocess.py       # Create training sequences (~5 min)
python 04_train.py            # Train the model (~1-2 hrs CPU)
python 05_build_faiss.py      # Build FAISS vector index (~2 min)
python 06_search_map.py       # Build search map for app (~5 min)
python 06_evaluate.py         # Evaluate the model (~5 min)
python app.py                 # Launch web app
```

Open browser at: **http://127.0.0.1:5000**

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Deep Learning | PyTorch |
| Vector Search | FAISS |
| Data Processing | Pandas, PyArrow |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript |

---

## 💡 Example Use Cases

| User Session | System Understands | Recommends |
|---|---|---|
| Mic → Ring Light → Webcam | Content Creator | UPS System, High-Performance RAM |
| Camera → Lens → Tripod | Photographer | Camera bag, Memory card, Filters |
| Phone → Earbuds → Case | Mobile user | Charger, Screen protector, Power bank |
| Laptop → Mouse → Keyboard | Office/Student | Monitor, USB hub, Webcam |

---

## 📝 Summary

> *"My project uses **Deep Learning** to understand a user's journey. By using a **Sequential Two-Tower model**, it can predict what a user needs next out of **1 million products** by analyzing the pattern of their previous 'Likes' in real-time."*
