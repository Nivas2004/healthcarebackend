import os
import uvicorn
import numpy as np
import tempfile
import gdown
import sqlite3
import tensorflow as tf
import random
from datetime import datetime
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# --- REPORTLAB IMPORTS FOR PROFESSIONAL PDF ---
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import VerticalBarChart

# ==========================================
# 1. DATABASE SETUP (SQLite)
# ==========================================
DB_NAME = "deepgyn_records.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_email TEXT,
            name TEXT,
            age INTEGER,
            location TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 2. FRONTEND CODE (Unchanged)
# ==========================================

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepGynScan AI | Advanced Diagnostics</title>
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
            margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow: hidden; background: #000; color: white;
        }

        #bg { position: fixed; top: 0; left: 0; z-index: -1; }

        /* --- LOGIN OVERLAY --- */
        #login-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.95); z-index: 2000;
            display: flex; justify-content: center; align-items: center;
        }
        .login-box {
            background: var(--glass-strong); padding: 40px; border-radius: 20px;
            border: 1px solid var(--primary); text-align: center; width: 350px;
            box-shadow: 0 0 50px rgba(0, 210, 255, 0.2);
        }

        /* --- NAVIGATION --- */
        nav {
            position: fixed; top: 0; width: 100%; padding: 20px 40px;
            display: flex; justify-content: space-between; align-items: center;
            z-index: 100; box-sizing: border-box; display: none;
        }
        .logo { font-size: 1.5rem; font-weight: 300; letter-spacing: 2px; text-transform: uppercase; }
        .logo span { font-weight: 800; color: var(--primary); }
        .nav-links button {
            background: transparent; border: none; color: #ccc; margin-left: 20px;
            font-size: 1rem; cursor: pointer; transition: 0.3s;
            text-transform: uppercase; letter-spacing: 1px;
        }
        .nav-links button:hover, .nav-links button.active { color: var(--primary); }

        /* --- CONTAINER --- */
        .container {
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            width: 90%; max-width: 1000px; height: 80vh;
            display: flex; justify-content: center; align-items: center;
            perspective: 1000px; display: none;
        }

        section { display: none; width: 100%; animation: fadeIn 0.6s ease-out forwards; }
        section.active-section { display: block; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

        /* --- UI ELEMENTS --- */
        .home-content { display: flex; gap: 40px; align-items: center; }
        .hero-text { flex: 1; }
        .hero-text h1 { font-size: 3.5rem; line-height: 1.1; margin-bottom: 20px; }
        .info-cards { display: flex; gap: 20px; margin-bottom: 30px; }
        .info-card { background: var(--glass); padding: 15px; border-radius: 10px; border: 1px solid var(--border); flex: 1; }
        .info-card h4 { color: var(--primary); margin: 0 0 5px 0; }
        .info-card p { font-size: 0.85rem; color: #aaa; margin: 0; }
        
        .cta-btn {
            padding: 15px 40px; background: linear-gradient(90deg, var(--primary), var(--secondary));
            border: none; border-radius: 50px; color: white; font-size: 1.1rem; font-weight: bold;
            cursor: pointer; box-shadow: 0 0 20px rgba(0, 210, 255, 0.3); transition: 0.3s;
        }
        .cta-btn:hover { transform: scale(1.05); }

        .app-wrapper { max-width: 500px; margin: 0 auto; }
        .card {
            background: var(--glass-strong); backdrop-filter: blur(20px);
            border: 1px solid var(--border); border-radius: 20px; padding: 30px;
            box-shadow: 0 0 50px rgba(0, 0, 0, 0.5); max-height: 80vh; overflow-y: auto;
        }
        .card::-webkit-scrollbar { width: 6px; }
        .card::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

        .form-group { margin-bottom: 15px; }
        label { display: block; font-size: 0.8rem; color: #aaa; margin-bottom: 5px; }
        
        input {
            width: 100%; padding: 12px; background: rgba(255,255,255,0.05);
            border: 1px solid var(--border); border-radius: 8px; color: white;
            outline: none; box-sizing: border-box; transition: 0.3s;
        }
        input:focus { border-color: var(--primary); background: rgba(255,255,255,0.1); }

        #img-preview { width: 100%; height: 150px; object-fit: cover; border-radius: 8px; margin-top: 10px; display: none; border: 1px solid var(--primary); }

        .btn {
            width: 100%; padding: 15px; border: none; border-radius: 8px;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5); color: white;
            font-weight: bold; cursor: pointer; margin-top: 10px;
            text-transform: uppercase; letter-spacing: 1px; transition: 0.3s;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0, 210, 255, 0.3); }
        .btn-download { background: linear-gradient(90deg, #11998e, #38ef7d); margin-top: 15px; display: none;}
        .btn-email { background: transparent; border: 1px solid var(--primary); color: var(--primary); margin-top: 10px; display: none;}
        .btn-email:hover { background: rgba(0, 210, 255, 0.1); }

        #result-panel { display: none; text-align: center; }
        .status-badge { display: inline-block; padding: 10px 20px; border-radius: 50px; font-weight: bold; margin-bottom: 15px; font-size: 1.2rem; border: 2px solid currentColor; }
        
        .history-table { width: 100%; border-collapse: collapse; margin-top: 20px; color: #ddd; font-size: 0.9rem; }
        .history-table th { text-align: left; padding: 10px; border-bottom: 1px solid #444; color: var(--primary); }
        .history-table td { padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); }

        .loader { display: none; border: 3px solid rgba(255,255,255,0.1); border-top: 3px solid var(--primary); border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 20px auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .toggle-link { color: var(--primary); cursor: pointer; font-size: 0.9rem; text-decoration: underline; margin-top: 10px; display: block; }
        .class-bar-bg { width: 100%; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; margin-top:5px; }
        .class-bar-fill { height: 100%; border-radius: 2px; transition: width 1s ease-out; }
        .class-header { display: flex; justify-content: space-between; font-size: 0.85rem; margin-top: 10px;}
    </style>
</head>
<body>

    <canvas id="bg"></canvas>

    <!-- 1. FIREBASE LOGIN OVERLAY -->
    <div id="login-overlay">
        <div class="login-box">
            <h2 style="color:white; margin-bottom:20px;" id="auth-title">DOCTOR LOGIN</h2>
            <div class="form-group">
                <input type="email" id="auth-email" placeholder="Email Address">
            </div>
            <div class="form-group">
                <input type="password" id="auth-password" placeholder="Password">
            </div>
            <button class="btn" id="auth-btn">Login</button>
            <p id="auth-error" style="color:var(--danger); font-size:0.8rem; margin-top:10px;"></p>
            <span class="toggle-link" id="toggle-auth-mode">Create new account</span>
        </div>
    </div>

    <!-- 2. NAVIGATION -->
    <nav id="main-nav">
        <div class="logo">DeepGynScan <span>AI</span></div>
        <div class="nav-links">
            <button onclick="switchSection('home')" id="nav-home" class="active">Home</button>
            <button onclick="switchSection('app')" id="nav-app">AI Analysis</button>
            <button onclick="loadHistory(); switchSection('history')" id="nav-history">Patient Records</button>
            <button onclick="logout()" style="color: var(--danger); border: 1px solid var(--danger); border-radius: 5px; padding: 5px 15px;">Logout</button>
        </div>
    </nav>

    <!-- 3. MAIN CONTAINER -->
    <div class="container" id="main-container">
        
        <!-- HOME SECTION -->
        <section id="home-section" class="active-section">
            <div class="home-content">
                <div class="hero-text">
                    <h1>Next-Gen <br>Cervical Diagnostics.</h1>
                    <p>
                        An AI-powered decision support system designed for pathologists. 
                        Features real-time cellular classification, probability breakdown, 
                        and automated reporting database.
                    </p>
                    <div class="info-cards">
                        <div class="info-card"><h4>Precision AI</h4><p>5-Class Detection Model</p></div>
                        <div class="info-card"><h4>Private Data</h4><p>User-Isolated Records</p></div>
                        <div class="info-card"><h4>Fast Reporting</h4><p>Instant PDF Generation</p></div>
                    </div>
                    <button class="cta-btn" onclick="switchSection('app')">Start New Analysis</button>
                </div>
            </div>
        </section>

        <!-- APP SECTION -->
        <section id="app-section">
            <div class="app-wrapper">
                <!-- Input Form -->
                <div id="input-panel" class="card">
                    <h2 style="text-align:center; margin-top:0;">Patient Analysis</h2>
                    
                    <form id="scanForm">
                        <div class="form-group">
                            <label>Patient Name</label>
                            <input type="text" id="pName" required placeholder="Ex: Jane Doe">
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
                            <label>Upload Cell Micrograph</label>
                            <input type="file" id="pFile" accept="image/*" required onchange="previewImage(this)">
                            <img id="img-preview" src="">
                        </div>

                        <button type="submit" class="btn" id="analyzeBtn">Analyze Sample</button>
                    </form>
                    <div class="loader" id="loader"></div>
                </div>

                <!-- Result Panel -->
                <div id="result-panel" class="card">
                    <h2 style="text-align:center; margin-top:0;">Diagnosis Report</h2>
                    <div id="statusText" class="status-badge" style="color: white;">Analyzing...</div>
                    
                    <div style="margin-top: 20px; text-align: left; background: rgba(255, 255, 255, 0.03); padding: 15px; border-radius: 10px;">
                        <div style="font-size:0.8rem; color:#aaa; margin-bottom:10px; border-bottom:1px solid #444;">AI CONFIDENCE BREAKDOWN</div>
                        <div id="details-list"></div>
                    </div>

                    <button class="btn btn-download" id="downloadBtn">Download PDF Report</button>
                    <button class="btn btn-email" id="emailBtn">Email Report</button>
                    <button class="btn" style="background:transparent; border:1px solid #555; margin-top:10px" onclick="resetScan()">New Scan</button>
                </div>
            </div>
        </section>

        <!-- HISTORY SECTION -->
        <section id="history-section">
            <div class="card" style="max-width: 800px; margin: 0 auto;">
                <h2 style="text-align:center;">Patient Records (My Patients)</h2>
                <div style="overflow-x:auto;">
                    <table class="history-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Date</th>
                                <th>Name</th>
                                <th>Diagnosis</th>
                                <th>Confidence</th>
                            </tr>
                        </thead>
                        <tbody id="history-table-body">
                            <!-- JS Injects Rows Here -->
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

    </div>

    <!-- Import Three.js & Firebase -->
    <script type="importmap">
      {
        "imports": {
          "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
          "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/",
          "firebase/app": "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js",
          "firebase/auth": "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js"
        }
      }
    </script>

    <!-- FIREBASE & AUTH LOGIC -->
    <script type="module">
        import { initializeApp } from "firebase/app";
        import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut, onAuthStateChanged } from "firebase/auth";

        // ------------------------------------------------------------------
        // REPLACE WITH YOUR FIREBASE CONFIG
        // ------------------------------------------------------------------
        const firebaseConfig = {
          apiKey:  "AIzaSyCc6SLKLQ61UaCe9KPHD1iIs4oD0w0W5Kc",
          authDomain: "healthcare-f573a.firebaseapp.com",
          projectId: "healthcare-f573a",
          storageBucket: "healthcare-f573a.firebasestorage.app",
          messagingSenderId: "380575106412",
          appId: "1:380575106412:web:3385b6d0060b88a6ed1f49"
        };
        // ------------------------------------------------------------------

        // Initialize Firebase
        let app, auth;
        try {
            app = initializeApp(firebaseConfig);
            auth = getAuth(app);
        } catch(e) {
            console.error("Firebase Config Error", e);
        }

        const authEmailInput = document.getElementById('auth-email');
        const authPassInput = document.getElementById('auth-password');
        const authBtn = document.getElementById('auth-btn');
        const authError = document.getElementById('auth-error');
        const toggleLink = document.getElementById('toggle-auth-mode');
        const authTitle = document.getElementById('auth-title');
        
        let isLoginMode = true;

        // Toggle Login / Register
        toggleLink.addEventListener('click', () => {
            isLoginMode = !isLoginMode;
            authTitle.innerText = isLoginMode ? "DOCTOR LOGIN" : "REGISTER ACCOUNT";
            authBtn.innerText = isLoginMode ? "Login" : "Register";
            toggleLink.innerText = isLoginMode ? "Create new account" : "Back to login";
            authError.innerText = "";
        });

        // Handle Auth Button Click
        authBtn.addEventListener('click', async () => {
            const email = authEmailInput.value;
            const password = authPassInput.value;
            authError.innerText = "";
            try {
                if(isLoginMode) await signInWithEmailAndPassword(auth, email, password);
                else await createUserWithEmailAndPassword(auth, email, password);
            } catch (error) {
                authError.innerText = error.message;
            }
        });

        // Auth State Observer
        onAuthStateChanged(auth, (user) => {
            if (user) {
                // Set global user email for backend identification
                window.currentUserEmail = user.email; 
                
                document.getElementById('login-overlay').style.display = 'none';
                document.getElementById('main-nav').style.display = 'flex';
                document.getElementById('main-container').style.display = 'flex';
                if(window.moveCamera) window.moveCamera('home');
            } else {
                window.currentUserEmail = null;
                document.getElementById('login-overlay').style.display = 'flex';
                document.getElementById('main-nav').style.display = 'none';
                document.getElementById('main-container').style.display = 'none';
            }
        });

        window.logout = () => signOut(auth);
    </script>

    <!-- 3D LOGIC & APP LOGIC -->
    <script type="module">
        import * as THREE from 'three';
        
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x000000, 0.05);
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 5;
        const renderer = new THREE.WebGLRenderer({ canvas: document.querySelector('#bg'), antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(window.innerWidth, window.innerHeight);

        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);
        const pointLight = new THREE.PointLight(0x00d2ff, 2, 50);
        pointLight.position.set(5, 5, 5);
        scene.add(pointLight);
        const rimLight = new THREE.PointLight(0x00d2ff, 1, 50);
        rimLight.position.set(-5, -5, 5);
        scene.add(rimLight);

        const geometry = new THREE.IcosahedronGeometry(2, 2); 
        const material = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.3, metalness: 0.8, wireframe: false });
        const cellMesh = new THREE.Mesh(geometry, material);
        scene.add(cellMesh);
        const wireGeo = new THREE.WireframeGeometry(geometry);
        const wireMat = new THREE.LineBasicMaterial({ color: 0x00d2ff, transparent: true, opacity: 0.1 });
        const wireframe = new THREE.LineSegments(wireGeo, wireMat);
        cellMesh.add(wireframe);

        const particlesGeo = new THREE.BufferGeometry();
        const particlesCount = 600;
        const posArray = new Float32Array(particlesCount * 3);
        for(let i=0; i<particlesCount*3; i++) posArray[i] = (Math.random() - 0.5) * 20;
        particlesGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
        const particlesMat = new THREE.PointsMaterial({ size: 0.05, color: 0xffffff, transparent: true, opacity: 0.5 });
        const particles = new THREE.Points(particlesGeo, particlesMat);
        scene.add(particles);

        function animate() {
            requestAnimationFrame(animate);
            cellMesh.rotation.x += 0.002;
            cellMesh.rotation.y += 0.003;
            particles.rotation.y -= 0.001;
            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

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
    </script>

    <script>
        const BASE_URL = ""; 
        let currentResultData = null; 

        function previewImage(input) {
            const preview = document.getElementById('img-preview');
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                }
                reader.readAsDataURL(input.files[0]);
            }
        }

        function switchSection(sectionId) {
            document.querySelectorAll('.nav-links button').forEach(b => b.classList.remove('active'));
            document.getElementById('nav-' + sectionId).classList.add('active');
            document.querySelectorAll('section').forEach(s => s.classList.remove('active-section'));
            document.getElementById(sectionId + '-section').classList.add('active-section');
            if(window.moveCamera) window.moveCamera(sectionId);
        }

        function resetScan() {
            document.getElementById('pName').value = "";
            document.getElementById('pAge').value = "";
            document.getElementById('pLoc').value = "";
            document.getElementById('pFile').value = "";
            document.getElementById('img-preview').style.display = 'none';
            document.getElementById('img-preview').src = "";
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('analyzeBtn').innerText = "Analyze Sample";
            document.getElementById('loader').style.display = "none";
            document.getElementById('result-panel').style.display = "none";
            document.getElementById('input-panel').style.display = "block";
            if(window.updateSceneColor) window.updateSceneColor("#00d2ff");
        }

        const form = document.getElementById('scanForm');
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            document.getElementById('analyzeBtn').disabled = true;
            document.getElementById('analyzeBtn').innerText = "Processing...";
            document.getElementById('loader').style.display = "block";

            const formData = new FormData();
            formData.append("file", document.getElementById('pFile').files[0]);

            try {
                const response = await fetch(`${BASE_URL}/predict`, { method: 'POST', body: formData });
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

                // Save with User Email
                if(window.currentUserEmail) {
                    await fetch(`${BASE_URL}/save-scan`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            doctor_email: window.currentUserEmail, 
                            name: currentResultData.patientName,
                            age: currentResultData.patientAge,
                            location: currentResultData.patientLocation,
                            prediction: currentResultData.prediction,
                            confidence: currentResultData.confidence
                        })
                    });
                }

                showResults(data);

            } catch (error) {
                alert("Error: " + error.message);
                document.getElementById('analyzeBtn').disabled = false;
                document.getElementById('analyzeBtn').innerText = "Analyze Sample";
                document.getElementById('loader').style.display = "none";
            }
        });

        function showResults(data) {
            document.getElementById('input-panel').style.display = "none";
            document.getElementById('result-panel').style.display = "block";

            const category = data.prediction; 
            let mainColor = "#00d2ff"; 
            if (category.includes("High Risk") || category.includes("Dyskeratotic")) mainColor = "#ff4b4b"; 
            else if (category.includes("Pre-cancerous")) mainColor = "#ffb347"; 
            else if (category.includes("Normal")) mainColor = "#00ff9d"; 

            const statusBadge = document.getElementById('statusText');
            statusBadge.innerText = category;
            statusBadge.style.color = mainColor;
            statusBadge.style.borderColor = mainColor;

            if(window.updateSceneColor) window.updateSceneColor(mainColor);

            const detailsList = document.getElementById('details-list');
            detailsList.innerHTML = ""; 
            const entries = Object.entries(data.mapped_details).sort((a, b) => b[1].confidence - a[1].confidence);

            entries.forEach(([clsName, info]) => {
                const percentage = (info.confidence * 100).toFixed(2);
                let barColor = "#00ff9d";
                if (info.category.includes("High Risk")) barColor = "#ff4b4b";
                else if (info.category.includes("Pre-cancerous")) barColor = "#ffb347";
                
                const div = document.createElement('div');
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

            document.getElementById('downloadBtn').style.display = "block";
            document.getElementById('emailBtn').style.display = "block";
            
            document.getElementById('emailBtn').onclick = () => {
                 const subject = `DeepGynScan Report: ${currentResultData.patientName}`;
                 const body = `Patient: ${currentResultData.patientName}%0D%0AAge: ${currentResultData.patientAge}%0D%0ADiagnosis: ${currentResultData.prediction}%0D%0AConfidence: ${(currentResultData.confidence * 100).toFixed(2)}%`;
                 window.location.href = `mailto:?subject=${subject}&body=${body}`;
            };
        }

        document.getElementById('downloadBtn').addEventListener('click', async () => {
            if(!currentResultData) return;
            const btn = document.getElementById('downloadBtn');
            btn.innerText = "Generating PDF...";
            try {
                const response = await fetch(`${BASE_URL}/generate-report`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentResultData)
                });
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none'; a.href = url;
                a.download = `Report_${currentResultData.patientName.replace(" ", "_")}.pdf`;
                document.body.appendChild(a); a.click();
                window.URL.revokeObjectURL(url);
                btn.innerText = "Download PDF Report";
            } catch (error) {
                alert("Error");
            }
        });

        async function loadHistory() {
            if(!window.currentUserEmail) return;
            // Send email as query param to filter history
            const res = await fetch(`${BASE_URL}/history?email=${encodeURIComponent(window.currentUserEmail)}`);
            const data = await res.json();
            const tbody = document.getElementById('history-table-body');
            tbody.innerHTML = "";
            
            if(data.history.length === 0) {
                tbody.innerHTML = "<tr><td colspan='5' style='text-align:center; color:#888;'>No records found for your account.</td></tr>";
                return;
            }

            data.history.forEach(row => {
                const tr = document.createElement('tr');
                let color = "#00ff9d";
                if(row.prediction.includes("High")) color = "#ff4b4b";
                else if(row.prediction.includes("Pre")) color = "#ffb347";

                tr.innerHTML = `
                    <td>#${row.id}</td>
                    <td style="font-size: 0.8rem; color:#888;">${row.timestamp}</td>
                    <td>${row.name}</td>
                    <td style="color: ${color}; font-weight: bold;">${row.prediction}</td>
                    <td>${(row.confidence * 100).toFixed(1)}%</td>
                `;
                tbody.appendChild(tr);
            });
        }
    </script>
</body>
</html>
"""

# ==========================================
# 3. BACKEND LOGIC
# ==========================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.h5")
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("📥 Model not found. Downloading...")
    url = "https://drive.google.com/uc?id=1L84L6Wiy9_SCLjgdPvnBgoQH8VRCsG4v"
    gdown.download(url, MODEL_PATH, quiet=False)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# --- Constants ---
classes = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"]
category_map = {
    "im_Dyskeratotic": "High Risk / Cancerous",
    "im_Koilocytotic": "Pre-cancerous",
    "im_Metaplastic": "Pre-cancerous",
    "im_Parabasal": "Normal",
    "im_Superficial-Intermediate": "Normal"
}

# --- Pydantic Models ---
class ScanData(BaseModel):
    doctor_email: str
    name: str
    age: int
    location: str
    prediction: str
    confidence: float

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def home():
    return html_content

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None: return {"error": "Model not loaded."}
    try:
        img = Image.open(file.file).resize((224, 224))
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)
        preds = model.predict(arr)[0]

        result = dict(zip(classes, preds.tolist()))
        predicted_class = classes[np.argmax(preds)]
        predicted_category = category_map[predicted_class]

        mapped_details = {
            cls: {"category": category_map.get(cls, "Unknown"), "confidence": float(score)}
            for cls, score in result.items()
        }

        return {
            "prediction": predicted_category,
            "confidence": float(np.max(preds)),
            "details": result,
            "mapped_details": mapped_details
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/save-scan")
async def save_scan(data: ScanData):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO scans (doctor_email, name, age, location, prediction, confidence, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (data.doctor_email, data.name, data.age, data.location, data.prediction, data.confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return {"status": "saved"}

@app.get("/history")
async def get_history(email: str):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM scans WHERE doctor_email = ? ORDER BY id DESC", (email,))
    rows = c.fetchall()
    conn.close()
    return {"history": [dict(row) for row in rows]}

# --- REPORT GENERATION (PROFESSIONAL REPORTLAB PDF) ---
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
        
        # --- PDF SETUP ---
        doc = SimpleDocTemplate(temp_file.name, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom Styles
        title_style = ParagraphStyle(name='Title', parent=styles['Heading1'], alignment=1, fontSize=18, spaceAfter=20)
        normal_style = styles['Normal']
        header_text_style = ParagraphStyle(name='Header', parent=styles['Normal'], alignment=2, fontSize=9)
        
        # --- 1. HEADER (Logo Placeholder + Lab Info) ---
        # Drawing a simple "Logo" using graphics since we can't rely on external images in a single file
        d = Drawing(100, 50)
        d.add(Rect(0, 0, 100, 50, fillColor=colors.HexColor('#00d2ff'), strokeColor=None))
        d.add(String(10, 20, "DEEPGYN", fontSize=14, fillColor=colors.white))
        d.add(String(10, 10, "SCAN AI", fontSize=8, fillColor=colors.white))
        
        # Table for Header (Logo Left, Text Right)
        header_data = [
            [d, Paragraph("<b>DEEPGYNSCAN DIAGNOSTICS LAB</b><br/>KPRIET, Coimbatore<br/>Licence No: 2764<br/>Phone: +91 8072568527", header_text_style)]
        ]
        header_table = Table(header_data, colWidths=[120, 400])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ALIGN', (1,0), (1,0), 'RIGHT'),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 20))
        
        # --- 2. TITLE ---
        elements.append(Paragraph("CERVICAL CYTOLOGY AI ANALYSIS", title_style))
        elements.append(Spacer(1, 10))
        
        # --- 3. PATIENT DEMOGRAPHICS (Grid) ---
        patient_data = [
            ["Patient Name:", patientName, "Patient ID:", "DG-" + str(random.randint(1000,9999))],
            ["Age / Gender:", f"{patientAge} Years / Female", "Date:", datetime.now().strftime("%Y-%m-%d")],
            ["Referred By:", "DeepGynScan AI", "Location:", patientLocation]
        ]
        
        t_patient = Table(patient_data, colWidths=[90, 180, 90, 170])
        t_patient.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
            ('BACKGROUND', (2,0), (2,-1), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('PADDING', (0,0), (-1,-1), 8),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ]))
        elements.append(t_patient)
        elements.append(Spacer(1, 25))
        
        # --- 4. DIAGNOSIS RESULT ---
        # Determine Color based on result
        res_color = "green"
        if "High" in prediction: res_color = "red"
        elif "Pre" in prediction: res_color = "orange"
        
        elements.append(Paragraph("<b>CLINICAL IMPRESSION:</b>", styles['Heading3']))
        result_html = f"<font size='14' color='{res_color}'><b>{prediction}</b></font>"
        elements.append(Paragraph(result_html, styles['Normal']))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"<b>AI Model Confidence:</b> {confidence*100:.2f}%", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # --- 5. CONFIDENCE BREAKDOWN TABLE ---
        elements.append(Paragraph("<b>Detailed Class Probabilities:</b>", styles['Heading4']))
        
        # Header for the table
        table_data = [["Class Name", "Risk Category", "Probability Score"]]
        
        # Sort details by score
        sorted_details = sorted(details.items(), key=lambda x: x[1], reverse=True)
        
        for cls, score in sorted_details:
            mapped_cls_category = category_map.get(cls, "Unknown")
            # Drawing a small bar for visualization inside table
            bar_width = int(score * 100)
            bar_color = colors.green
            if "High" in mapped_cls_category: bar_color = colors.red
            elif "Pre" in mapped_cls_category: bar_color = colors.orange
            
            # Simple text representation of bar for robustness
            prob_text = f"{score*100:.2f}%"
            table_data.append([cls.replace("im_", ""), mapped_cls_category, prob_text])
            
        t_conf = Table(table_data, colWidths=[200, 200, 100])
        t_conf.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00d2ff')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 10),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
        elements.append(t_conf)
        elements.append(Spacer(1, 40))
        
        # --- 6. FOOTER / SIGNATURE ---
        sig_data = [
            ["_______________________", "_______________________"],
            ["AI System Generated", "Chief Pathologist Signature"]
        ]
        t_sig = Table(sig_data, colWidths=[250, 250])
        t_sig.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.grey),
        ]))
        elements.append(t_sig)
        elements.append(Spacer(1, 10))
        
        disclaimer = "<b>DISCLAIMER:</b> This report is generated by an Artificial Intelligence system (DeepGynScan) and serves as a preliminary screening tool. It does not replace a professional medical diagnosis. All results must be clinically correlated and verified by a certified pathologist."
        elements.append(Paragraph(disclaimer, ParagraphStyle(name='Footer', parent=styles['Normal'], fontSize=7, textColor=colors.grey, alignment=4))) # Justify
        
        # BUILD PDF
        doc.build(elements)

        return FileResponse(temp_file.name, filename="DeepGynScan_Report.pdf", media_type="application/pdf")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)