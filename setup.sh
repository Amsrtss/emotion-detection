#!/bin/bash

# 1. Buat virtual environment
python3 -m venv venv

# 2. Aktifkan virtual environment
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install semua dependencies dari requirements.txt
pip install -r requirements.txt

# 5. Jalankan aplikasi Streamlit
streamlit run app.py
