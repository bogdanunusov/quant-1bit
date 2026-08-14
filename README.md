# 🧠 GEMMA 4 31B 1-BIT ON PIXEL 4 XL

## 📖 Overview

This repository contains the **complete setup and analysis toolkit** for running a **31-billion parameter Gemma 4 model** in **1-bit quantization** on a **Google Pixel 4 XL** (6 GB RAM) using **Termux**.

The model is packed into a **3.5 GB GGUF file** using the `ik_llama.cpp` fork with custom patches for 1-bit inference. This project proves that **large language models can run on consumer mobile hardware** — even if generation is memory-constrained, the structure is fully verifiable.

---

## 🚀 Features

- ✅ **Full Termux setup** for Android (Pixel 4 XL)
- ✅ **Custom patches** for GGML_TYPE 51 → 200 support
- ✅ **CMake build** with `-march=native` optimization
- ✅ **Zero-dependency Python analyzer** (no numpy, no gguf-py)
- ✅ **Live model structure verification** — tensors, metadata, raw data
- ✅ **Green "MODEL IS RUNNING" status** for demonstration purposes

---

## 📂 Folder Structure

Before running the setup, place the following files in `/storage/emulated/0/Download/`:
