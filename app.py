import os
import uvicorn
import numpy as np
import tempfile
import gdown
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ==========================================
# 1. FRONTEND CODE (Updated Name to DeepGynScan)
# ==========================================

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepGynScan AI | Cervical Cancer Detection</title>
    <style>
        :root {
            --primary: #00d2ff;
            --secondary: #3a7bd5;
            --danger: #ff4b4b;
            --glass: rgba(16, 20, 30, 0.7);
            --glass-strong: rgba(10, 15, 25, 0.95);
            --border: rgba(255, 255, 255, 0.1);
        }

        body {
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow: hidden;
            background: #000;
            color: white;
        }

        /* 3D Canvas Background */
        #bg {
            position: fixed;
            top: 0;
            left: 0;
            z-index: -1;
        }

        /* --- NAVIGATION --- */
        nav {
            position: fixed;
            top: 0;
            width: 100%;
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 100;
            box-sizing: border-box;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: 300;
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        .logo span { font-weight: 800; color: var(--primary); }

        .nav-links button {
            background: transparent;
            border: none;
            color: #ccc;
            margin-left: 20px;
            font-size: 1rem;
            cursor: pointer;
            transition: 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .nav-links button:hover, .nav-links button.active {
            color: var(--primary);
        }

        /* --- MAIN CONTAINER --- */
        .container {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90%;
            max-width: 1000px;
            height: 80vh;
            display: flex;
            justify-content: center;
            align-items: center;
            perspective: 1000px;
        }

        /* --- SECTIONS --- */
        section {
            display: none; /* Hidden by default */
            width: 100%;
            animation: fadeIn 0.6s ease-out forwards;
        }
        
        section.active-section {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* --- HOME SECTION STYLES --- */
        .home-content {
            display: flex;
            gap: 40px;
            align-items: center;
        }

        .hero-text {
            flex: 1;
        }

        .hero-text h1 {
            font-size: 3.5rem;
            line-height: 1.1;
            margin-bottom: 20px;
            background: linear-gradient(90deg, #fff, #aaa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-text p {
            font-size: 1.1rem;
            line-height: 1.6;
            color: #ddd;
            margin-bottom: 30px;
        }

        .info-cards {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }

        .info-card {
            background: var(--glass);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid var(--border);
            flex: 1;
        }
        .info-card h4 { color: var(--primary); margin: 0 0 5px 0; }
        .info-card p { font-size: 0.85rem; color: #aaa; margin: 0; }

        .cta-btn {
            padding: 15px 40px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            border: none;
            border-radius: 50px;
            color: white;
            font-size: 1.1rem;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 0 20px rgba(0, 210, 255, 0.3);
            transition: 0.3s;
        }
        .cta-btn:hover { transform: scale(1.05); box-shadow: 0 0 30px rgba(0, 210, 255, 0.6); }

        /* --- APP/UPLOAD SECTION STYLES --- */
        .app-wrapper {
            max-width: 500px;
            margin: 0 auto;
        }

        .card {
            background: var(--glass-strong);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 0 50px rgba(0, 0, 0, 0.5);
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .card::-webkit-scrollbar { width: 6px; }
        .card::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

        .form-group { margin-bottom: 15px; }
        label { display: block; font-size: 0.8rem; color: #aaa; margin-bottom: 5px; }
        
        input[type="text"], input[type="number"], input[type="file"] {
            width: 100%;
            padding: 12px;
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: white;
            outline: none;
            box-sizing: border-box;
            transition: 0.3s;
        }
        input:focus { border-color: var(--primary); background: rgba(255,255,255,0.1); }

        .btn {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 8px;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            color: white;
            font-weight: bold;
            cursor: pointer;
            margin-top: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: 0.3s;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0, 210, 255, 0.3); }
        .btn:disabled { opacity: 0.6; cursor: wait; }
        .btn-download { background: linear-gradient(90deg, #11998e, #38ef7d); margin-top: 15px; display: none;}

        /* Results */
        #result-panel { display: none; text-align: center; }
        .status-badge {
            display: inline-block; padding: 10px 20px; border-radius: 50px;
            font-weight: bold; margin-bottom: 15px; font-size: 1.2rem; border: 2px solid currentColor;
        }
        
        /* Detailed Breakdown Styles */
        .breakdown-container {
            margin-top: 20px;
            text-align: left;
            background: rgba(255, 255, 255, 0.03);
            padding: 15px;
            border-radius: 10px;
        }
        .breakdown-title {
            font-size: 0.9rem;
            color: #aaa;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 1px solid #333;
            padding-bottom: 5px;
        }
        .class-item {
            margin-bottom: 12px;
        }
        .class-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 3px;
        }
        .class-bar-bg {
            width: 100%;
            height: 4px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
        }
        .class-bar-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 1s ease-out;
        }

        .loader {
            display: none; border: 3px solid rgba(255,255,255,0.1); border-top: 3px solid var(--primary);
            border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 20px auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>

    <canvas id="bg"></canvas>

    <!-- Navigation -->
    <nav>
        <!-- RENAMED HERE -->
        <div class="logo">DeepGynScan <span>AI</span></div>
        <div class="nav-links">
            <button onclick="switchSection('home')" id="nav-home" class="active">Home Awareness</button>
            <button onclick="switchSection('app')" id="nav-app">AI Analysis</button>
        </div>
    </nav>

    <div class="container">
        
        <!-- SECTION 1: HOME / AWARENESS -->
        <section id="home-section" class="active-section">
            <div class="home-content">
                <div class="hero-text">
                    <h1>Early Detection <br>Saves Lives.</h1>
                    <p>
                        Cervical cancer is one of the most preventable types of cancer. 
                        Regular screening and early diagnosis are crucial. Our AI-powered tool assists 
                        pathologists in identifying abnormal cells with high precision.
                    </p>
                    
                    <div class="info-cards">
                        <div class="info-card">
                            <h4>4th Most Common</h4>
                            <p>Cancer in women globally (WHO).</p>
                        </div>
                        <div class="info-card">
                            <h4>99% Treatable</h4>
                            <p>When detected in pre-cancerous stages.</p>
                        </div>
                        <div class="info-card">
                            <h4>AI Precision</h4>
                            <p>Deep Learning analysis of Pap smears.</p>
                        </div>
                    </div>

                    <button class="cta-btn" onclick="switchSection('app')">Start Screening Now</button>
                </div>
            </div>
        </section>

        <!-- SECTION 2: APP / UPLOAD -->
        <section id="app-section">
            <div class="app-wrapper">
                <!-- Input Form -->
                <div id="input-panel" class="card">
                    <h2 style="text-align:center; margin-top:0;">Upload Sample</h2>
                    <p style="text-align:center; color:#888; font-size:0.9rem; margin-bottom:20px;">
                        Upload a cell micrograph for classification
                    </p>
                    
                    <form id="scanForm">
                        <div class="form-group">
                            <label>Patient Name</label>
                            <input type="text" id="pName" required placeholder="Jane Doe">
                        </div>
                        <div class="form-group" style="display:flex; gap:10px;">
                            <div style="flex:1">
                                <label>Age</label>
                                <input type="number" id="pAge" required placeholder="35">
                            </div>
                            <div style="flex:2">
                                <label>Location</label>
                                <input type="text" id="pLoc" required placeholder="New York, USA">
                            </div>
                        </div>
                        <div class="form-group">
                            <label>Micrograph Image</label>
                            <input type="file" id="pFile" accept="image/*" required>
                        </div>

                        <button type="submit" class="btn" id="analyzeBtn">Analyze Sample</button>
                    </form>
                    <div class="loader" id="loader"></div>
                </div>

                <!-- Result Panel -->
                <div id="result-panel" class="card">
                    <h2 style="text-align:center; margin-top:0;">Analysis Complete</h2>
                    
                    <div id="statusText" class="status-badge" style="color: white;">Analyzing...</div>
                    
                    <!-- Detailed Breakdown Section -->
                    <div class="breakdown-container">
                        <div class="breakdown-title">Class Probability Breakdown</div>
                        <div id="details-list">
                            <!-- JS will inject rows here -->
                        </div>
                    </div>

                    <button class="btn btn-download" id="downloadBtn">Download PDF Report</button>
                    <button class="btn" style="background:transparent; border:1px solid #555; margin-top:10px" onclick="location.reload()">New Scan</button>
                </div>
            </div>
        </section>

    </div>

    <!-- Import Three.js -->
    <script type="importmap">
      {
        "imports": {
          "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
          "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }
      }
    </script>

    <script type="module">
        import * as THREE from 'three';
        
        // --- 1. 3D SCENE SETUP ---
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x000000, 0.05);

        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 5;

        const renderer = new THREE.WebGLRenderer({ canvas: document.querySelector('#bg'), antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(window.innerWidth, window.innerHeight);

        // Lights
        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);
        
        const pointLight = new THREE.PointLight(0x00d2ff, 2, 50);
        pointLight.position.set(5, 5, 5);
        scene.add(pointLight);
        
        const rimLight = new THREE.PointLight(0x00d2ff, 1, 50);
        rimLight.position.set(-5, -5, 5);
        scene.add(rimLight);

        // The "Cell" Object
        const geometry = new THREE.IcosahedronGeometry(2, 2); 
        const material = new THREE.MeshStandardMaterial({ 
            color: 0x111111, roughness: 0.3, metalness: 0.8, wireframe: false
        });
        const cellMesh = new THREE.Mesh(geometry, material);
        scene.add(cellMesh);

        // Wireframe overlay
        const wireGeo = new THREE.WireframeGeometry(geometry);
        const wireMat = new THREE.LineBasicMaterial({ color: 0x00d2ff, transparent: true, opacity: 0.1 });
        const wireframe = new THREE.LineSegments(wireGeo, wireMat);
        cellMesh.add(wireframe);

        // Particles
        const particlesGeo = new THREE.BufferGeometry();
        const particlesCount = 600;
        const posArray = new Float32Array(particlesCount * 3);
        for(let i=0; i<particlesCount*3; i++) posArray[i] = (Math.random() - 0.5) * 20;
        particlesGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
        const particlesMat = new THREE.PointsMaterial({ size: 0.05, color: 0xffffff, transparent: true, opacity: 0.5 });
        const particles = new THREE.Points(particlesGeo, particlesMat);
        scene.add(particles);

        // Animation Loop
        function animate() {
            requestAnimationFrame(animate);
            cellMesh.rotation.x += 0.002;
            cellMesh.rotation.y += 0.003;
            particles.rotation.y -= 0.001;
            renderer.render(scene, camera);
        }
        animate();

        // Handle Resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // Function exposed to change color
        window.updateSceneColor = (colorHex) => {
            const color = new THREE.Color(colorHex);
            pointLight.color.set(color);
            rimLight.color.set(color);
            wireMat.color.set(color);
        };
        
        window.moveCamera = (section) => {
            if(section === 'app') cellMesh.position.x = -2.2;
            else cellMesh.position.x = 2.5; 
        }
        
        window.moveCamera('home');
    </script>

    <script>
        // --- 2. UI LOGIC ---
        function switchSection(sectionId) {
            document.querySelectorAll('.nav-links button').forEach(b => b.classList.remove('active'));
            document.getElementById('nav-' + sectionId).classList.add('active');
            document.querySelectorAll('section').forEach(s => s.classList.remove('active-section'));
            document.getElementById(sectionId + '-section').classList.add('active-section');
            if(window.moveCamera) window.moveCamera(sectionId);
        }

        // --- 3. BACKEND INTEGRATION ---
        const BASE_URL = ""; 
        let currentResultData = null; 

        const form = document.getElementById('scanForm');
        const loader = document.getElementById('loader');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const inputPanel = document.getElementById('input-panel');
        const resultPanel = document.getElementById('result-panel');
        const downloadBtn = document.getElementById('downloadBtn');
        const detailsList = document.getElementById('details-list');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            analyzeBtn.disabled = true;
            analyzeBtn.innerText = "Processing...";
            loader.style.display = "block";

            const formData = new FormData();
            formData.append("file", document.getElementById('pFile').files[0]);

            try {
                const response = await fetch(`${BASE_URL}/predict`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error("Backend connection failed");

                const data = await response.json();
                
                currentResultData = {
                    prediction: data.prediction,
                    confidence: data.confidence,
                    details: data.details,
                    patientName: document.getElementById('pName').value,
                    patientAge: parseInt(document.getElementById('pAge').value),
                    patientLocation: document.getElementById('pLoc').value
                };

                showResults(data);

            } catch (error) {
                alert("Error: " + error.message);
                analyzeBtn.disabled = false;
                analyzeBtn.innerText = "Analyze Sample";
                loader.style.display = "none";
            }
        });

        function showResults(data) {
            inputPanel.style.display = "none";
            resultPanel.style.display = "block";

            const category = data.prediction; 
            
            // Set Main Badge Color
            let mainColor = "#00d2ff"; 
            if (category.includes("High Risk") || category.includes("Dyskeratotic")) mainColor = "#ff4b4b"; // Red
            else if (category.includes("Pre-cancerous")) mainColor = "#ffb347"; // Orange
            else if (category.includes("Normal")) mainColor = "#00ff9d"; // Green

            const statusBadge = document.getElementById('statusText');
            statusBadge.innerText = category;
            statusBadge.style.color = mainColor;
            statusBadge.style.borderColor = mainColor;

            // Update 3D Light
            if(window.updateSceneColor) window.updateSceneColor(mainColor);

            // --- GENERATE ALL 5 PROGRESS BARS ---
            detailsList.innerHTML = ""; // Clear old
            
            // Convert mapped_details to array and sort by confidence descending
            const entries = Object.entries(data.mapped_details).sort((a, b) => b[1].confidence - a[1].confidence);

            entries.forEach(([clsName, info]) => {
                const percentage = (info.confidence * 100).toFixed(2);
                
                // Determine specific color for this bar
                let barColor = "#00ff9d"; // Default Normal
                if (info.category.includes("High Risk") || clsName.includes("Dyskeratotic")) barColor = "#ff4b4b";
                else if (info.category.includes("Pre-cancerous")) barColor = "#ffb347";
                
                const div = document.createElement('div');
                div.className = 'class-item';
                div.innerHTML = `
                    <div class="class-header">
                        <span style="color:${barColor}">${clsName.replace('im_', '')}</span>
                        <span>${percentage}%</span>
                    </div>
                    <div class="class-bar-bg">
                        <div class="class-bar-fill" style="width: ${percentage}%; background-color: ${barColor};"></div>
                    </div>
                `;
                detailsList.appendChild(div);
            });

            downloadBtn.style.display = "block";
        }

        downloadBtn.addEventListener('click', async () => {
            if(!currentResultData) return;
            downloadBtn.innerText = "Generating PDF...";
            downloadBtn.disabled = true;
            try {
                const response = await fetch(`${BASE_URL}/generate-report`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentResultData)
                });
                if (!response.ok) throw new Error("Report generation failed");
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none'; a.href = url;
                a.download = `Report_${currentResultData.patientName.replace(" ", "_")}.pdf`;
                document.body.appendChild(a); a.click();
                window.URL.revokeObjectURL(url);
                downloadBtn.innerText = "Download PDF Report";
                downloadBtn.disabled = false;
            } catch (error) {
                alert("Error generating report: " + error.message);
                downloadBtn.innerText = "Download Failed";
                downloadBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

# ==========================================
# 2. BACKEND LOGIC (FastAPI + Model)
# ==========================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading Logic ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("📥 Model not found. Downloading...")
    # Using your Google Drive ID
    url = "https://drive.google.com/uc?id=1L84L6Wiy9_SCLjgdPvnBgoQH8VRCsG4v"
    gdown.download(url, MODEL_PATH, quiet=False)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# --- Classes & Maps ---
classes = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate"
]

category_map = {
    "im_Dyskeratotic": "High Risk / Cancerous",
    "im_Koilocytotic": "Pre-cancerous",
    "im_Metaplastic": "Pre-cancerous",
    "im_Parabasal": "Normal",
    "im_Superficial-Intermediate": "Normal"
}

# --- Routes ---

# 1. SERVE HTML
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serves the 3D Frontend"""
    return html_content

# 2. PREDICT ENDPOINT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please check model file."}
    try:
        img = Image.open(file.file).resize((224, 224))
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)
        preds = model.predict(arr)[0]

        result = dict(zip(classes, preds.tolist()))
        predicted_class = classes[np.argmax(preds)]
        predicted_category = category_map[predicted_class]

        mapped_details = {
            cls: {
                "category": category_map.get(cls, "Unknown"),
                "confidence": float(score)
            }
            for cls, score in result.items()
        }

        return {
            "prediction": predicted_category,
            "confidence": float(np.max(preds)),
            "details": result,
            "mapped_details": mapped_details
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# 3. REPORT GENERATION ENDPOINT
@app.post("/generate-report")
async def generate_report(
    prediction: str = Body(...),
    confidence: float = Body(...),
    details: dict = Body(...),
    patientName: str = Body(...),
    patientAge: int = Body(...),
    patientLocation: str = Body(...)
):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        c = canvas.Canvas(temp_file.name, pagesize=letter)
        width, height = letter

        # Header
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(width / 2, 770, "🏥 Cervical Cancer Detection Report")
        c.setFont("Helvetica", 12)
        c.drawCentredString(width / 2, 750, "AI-Assisted Medical Report")

        # Patient Info
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 720, "Patient Information:")
        c.setFont("Helvetica", 12)
        c.drawString(70, 700, f"Name: {patientName}")
        c.drawString(70, 680, f"Age: {patientAge}")
        c.drawString(70, 660, f"Location: {patientLocation}")

        # Prediction Info
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 630, "Prediction Result:")
        c.setFont("Helvetica", 12)
        c.drawString(70, 610, f"Predicted Category: {prediction}")
        c.drawString(70, 590, f"Confidence: {confidence*100:.2f}%")

        # Confidence Breakdown
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 560, "Confidence Breakdown per Class:")
        y = 540
        c.setFont("Helvetica-Bold", 12)
        c.drawString(60, y, "Class")
        c.drawString(250, y, "Category")
        c.drawString(400, y, "Confidence")
        c.line(50, y-2, 500, y-2)
        y -= 20
        c.setFont("Helvetica", 12)
        for cls, score in details.items():
            mapped_cls_category = category_map.get(cls, "Unknown")
            c.drawString(60, y, cls)
            c.drawString(250, y, mapped_cls_category)
            c.drawString(400, y, f"{score*100:.2f}%")
            y -= 20

        # Footer
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, 100, "⚕️ Disclaimer: AI-assisted report, not a medical diagnosis.")
        c.drawString(50, 85, "Consult a certified doctor for professional advice.")

        c.showPage()
        c.save()

        return FileResponse(temp_file.name, filename="Cancer_Report.pdf", media_type="application/pdf")
    except Exception as e:
        return {"error": f"Report generation failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)