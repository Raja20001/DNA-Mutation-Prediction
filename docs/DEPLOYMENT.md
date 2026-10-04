# DNA-QBio: Publication Deployment Guide
## Hosting on GitHub Pages & Vercel for Academic Research Publication

The **DNA-QBio Intelligence Platform** features a universal dual-mode architecture designed for immediate, zero-friction publication on both **GitHub Pages** and **Vercel**. 

The platform runs **100% client-side** using hardware-accelerated Canvas/WebGL 3D graphics, in-silico variant scoring, Needleman-Wunsch sequence alignment, and Plotly.js charts—with **zero server-side daemon requirements**. It also includes a seamless bridge to connect to a live Python Flask REST API if local or cloud backend execution is desired.

---

## 🚀 Quick Decision Matrix

| Feature / Platform | GitHub Pages | Vercel | Local Flask App |
| :--- | :--- | :--- | :--- |
| **Hosting Type** | Static / Global CDN | Edge Network CDN | Localhost Daemon |
| **Setup Effort** | 1 Minute (Zero Config) | 1 Click Import | Python virtualenv |
| **Backend Required** | ❌ None (Client In-Silico) | ❌ None (Client In-Silico) | ✅ `python flask_app/app.py` |
| **3D DNA Helix & Bloch Sphere**| ✅ Fully Functional | ✅ Fully Functional | ✅ Fully Functional |
| **Multi-Model Inference & Alignment**| ✅ Client Engine | ✅ Client Engine | ✅ Full Python ML Models |
| **ACMG/ClinVar Evidence Engine** | ✅ Fully Functional | ✅ Fully Functional | ✅ Fully Functional |
| **Presentation Deck & Report** | ✅ Fully Functional | ✅ Fully Functional | ✅ Fully Functional |
| **Cost** | 100% Free | 100% Free (Hobby Tier)| Free (Self-Hosted) |

---

## 📦 Method 1: Deploy to GitHub Pages

GitHub Pages serves the static publication portal directly from your GitHub repository.

### Option 1A: Automated Deployment via GitHub Actions (Recommended)

This repository includes a pre-configured GitHub Actions workflow in [`.github/workflows/deploy-pages.yml`](../.github/workflows/deploy-pages.yml).

1. **Push your code** to GitHub:
   ```bash
   git add .
   git commit -m "feat: complete DNA-QBio publication web platform"
   git push origin main
   ```
2. On GitHub, navigate to your repository:
   - Click **Settings** (top tab).
   - In the left sidebar, click **Pages**.
   - Under **Build and deployment** → **Source**, change the dropdown from *Deploy from a branch* to **GitHub Actions**.
3. Go to the **Actions** tab in your repository:
   - You will see the workflow **"Deploy DNA-QBio Publication Portal to GitHub Pages"** running automatically.
   - Upon completion (~45 seconds), your live website URL will be displayed:
     ```text
     https://<your-username>.github.io/<repository-name>/
     ```

### Option 1B: Classic Branch Deployment (Settings GUI)

If you prefer not to use GitHub Actions:

1. Push your repository to GitHub.
2. Go to **Settings** → **Pages**.
3. Under **Build and deployment** → **Source**, select **Deploy from a branch**.
4. Set:
   - **Branch:** `main` (or `master`)
   - **Folder:** `/ (root)` or `/docs` (both contain the complete production `index.html` and `static/` assets)
5. Click **Save**. GitHub Pages will build and publish your site in 1–2 minutes.

---

## ⚡ Method 2: Deploy to Vercel

Vercel provides edge caching, automatic HTTPS, and instant preview deployments on every push.

### Option 2A: Via Vercel Web Dashboard (1-Click)

1. Sign in to [Vercel](https://vercel.com) using your GitHub account.
2. Click the **Add New...** button and select **Project**.
3. Locate your `DNA` repository and click **Import**.
4. In the **Configure Project** screen:
   - **Framework Preset:** Leave as *Other* (or *Vite/Static*).
   - **Root Directory:** `./`
   - **Build Command:** Leave blank (no compilation required).
   - **Output Directory:** Leave blank (Vercel automatically detects root `index.html` and [`vercel.json`](../vercel.json)).
5. Click **Deploy**.
6. Within 15 seconds, Vercel will generate your live production URL:
   ```text
   https://dna-qbio-<your-username>.vercel.app
   ```

### Option 2B: Via Vercel CLI

If you have Node.js installed:

```bash
# 1. Install Vercel CLI globally (if not already installed)
npm install -g vercel

# 2. Deploy directly from repository root
vercel

# 3. Deploy to production
vercel --prod
```

The CLI will read [`vercel.json`](../vercel.json) and output your live production URL immediately.

---

## 🔌 Dual-Mode Architecture: Client In-Silico vs. Live Python Backend

The publication portal features a built-in **Mode Switcher** in the top navigation bar:

1. **Default Mode: Client In-Silico Engine (GitHub Pages / Vercel)**
   - All 9 model architectures, base transition categorizations, Needleman-Wunsch alignments, and ACMG calculations execute locally in pure JavaScript.
   - Zero server requirements, zero latency, 100% offline uptime.
2. **Live Python Backend Mode (Optional)**
   - Click the green badge in the navigation bar: **"Client In-Silico Active"**.
   - Select **Live Python Backend API (Flask / Cloud)**.
   - Enter your backend URL:
     - Local machine: `http://localhost:5000`
     - Cloud deployed: `https://your-dna-api.onrender.com`
   - Click **Ping Connection** to verify, then click **Save & Apply**.
   - All subsequent analyses will query the live Qiskit 1.0+ Aer quantum simulator and PyTorch deep models on your backend server.

### Optional: Deploying the Python Flask Backend to the Cloud

If you want the live Python backend hosted publicly alongside your frontend:
- **Render.com / Fly.io / Railway / Heroku:**
  - Build command: `pip install -r requirements.txt`
  - Start command: `python flask_app/app.py` or `gunicorn flask_app.app:app`
  - Copy the resulting URL (e.g. `https://dna-backend.onrender.com`) and paste it into the frontend's API Configuration modal.

---

## 🧪 Post-Deployment Verification Checklist

Once your website is live on GitHub Pages or Vercel, verify all features:

- [ ] **Overview Tab:** 3D DNA Double Helix rotates smoothly, Pause button stops rotation, Mutation Pulse highlights red mutant base pairs, and speed slider adjusts rate.
- [ ] **Workflow Tab:** Clicking each of the 7 stages updates the Mathematical Formalism and displays live JSON execution contracts.
- [ ] **Quantum Lab Tab:** Dragging mouse rotates 3D Bloch sphere; clicking gates ($H, X, Y, Z, S, T$) updates quantum statevector angles ($\theta, \phi$) and basis probabilities in real-time.
- [ ] **Mutation Studio Tab:**
  - Selecting *Sample 1: TP53 R273H* auto-populates the sequence.
  - Clicking *Run Multi-Model Variant Analysis* features HQ-CMFN as the champion with 98.2% accuracy.
  - Expanding *Inspect All 9 Models Evaluated* displays the comparative leaderboard.
  - Clicking *Classify Base Transition* identifies Transition ($Ti$) with purine-purine retention impact.
  - Clicking *Localize Variant Coordinate* generates inline sequence highlighting.
  - Checking ACMG boxes updates the classification tier in real-time.
- [ ] **Benchmarks Tab:** Interactive Plotly.js Radar and ROC charts display model comparison curves.
- [ ] **Slide Deck Tab:** Previous/Next buttons and the slide dropdown navigate through all 25 defense slides.
- [ ] **Thesis Dossier Tab:** Clicking *Download Report (.md)* generates the markdown document.
