import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from google import genai
import time

# ==========================================
# 1. KONFIGURASI UTAMA HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Sistem Analisis Kesehatan Gigi AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. INISIALISASI SESSION STATE (LOGIKA LOGIN)
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'role' not in st.session_state:
    st.session_state.role = None

# ==========================================
# 3. CACHING MODEL (YOLO & ViT)
# ==========================================
@st.cache_resource 
def load_yolo_model():
    return YOLO("best8.pt")

@st.cache_resource
def load_vit_model():
    device = torch.device("cpu")
    class_names = ['Initial', 'Moderate', 'Normal', 'Severe']
    
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    num_features = model.head.in_features
    
    model.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, len(class_names))
    )
    
    try:
        model.load_state_dict(torch.load('vit_karies_final_model (2).pth', map_location=device))
        model.to(device)
        model.eval() 
        return model, class_names, device, True
    except Exception as e:
        return None, None, None, False

model_yolo = load_yolo_model()
model_inference, class_names, device, vit_model_loaded = load_vit_model()

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. FUNGSI DIAGNOSIS OFFLINE (FALLBACK)
# ==========================================
def get_offline_diagnosis_karies(result):
    if result == "Normal":
        return "**Deskripsi:** Gigi berada dalam kondisi sehat tanpa adanya tanda-tanda demineralisasi.\n\n**Rekomendasi:** Lanjutkan rutinitas menyikat gigi 2 kali sehari dengan pasta gigi ber-fluoride dan lakukan pemeriksaan rutin ke dokter gigi setiap 6 bulan sekali."
    elif result == "Initial":
        return "**Deskripsi:** Terdeteksi karies pada tahap awal (Lesi Dini). Proses demineralisasi mulai terjadi pada lapisan enamel luar.\n\n**Rekomendasi:** Tindakan non-invasif. Disarankan melakukan aplikasi Topikal Fluoride di dokter gigi untuk remineralisasi, serta kurangi konsumsi gula."
    elif result == "Moderate":
        return "**Deskripsi:** Karies telah mencapai tingkat sedang. Kerusakan enamel sudah membentuk lubang (kavitasi) dangkal.\n\n**Rekomendasi:** Memerlukan tindakan restoratif. Segera ke dokter gigi untuk pembersihan jaringan karies dan penambalan (Restorasi) agar lubang tidak semakin dalam."
    elif result == "Severe":
        return "**Deskripsi:** Karies pada tingkat sangat parah. Kerusakan telah mencapai lapisan dentin dalam dan mendekati ruang pulpa (saraf gigi).\n\n**Rekomendasi:** Memerlukan evaluasi radiografis (Rontgen). Tindakan medis yang direkomendasikan adalah Perawatan Saluran Akar (Endodontik) jika gigi masih bisa diselamatkan, atau pencabutan (Ekstraksi)."
    return "Data tidak dikenali."

def get_offline_diagnosis_kalkulus(num_boxes):
    if num_boxes > 0:
         return f"**Deskripsi:** Sistem mendeteksi adanya endapan mineral keras (kalkulus/karang gigi) pada {num_boxes} titik di permukaan gigi.\n\n**Rekomendasi:** Sangat disarankan untuk segera mengunjungi dokter gigi guna melakukan tindakan *Scaling* (pembersihan karang gigi)."
    else:
         return "**Deskripsi:** Tidak terdeteksi adanya penumpukan kalkulus (karang gigi) pada citra yang dianalisis.\n\n**Rekomendasi:** Tetap jaga kebersihan rongga mulut dengan menyikat gigi minimal 2 kali sehari secara menyeluruh."

# ==========================================
# PENGAMBILAN API KEY SECARA RAHASIA
# ==========================================
# Mengambil API Key dari .streamlit/secrets.toml
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    API_KEY = None

# ==========================================
# 5. HALAMAN LOGIN
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center;'>🔐 Portal Deteksi Kesehatan Gigi AI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_umum, tab_dokter = st.tabs(["👤 Masyarakat Umum", "🩺 Dokter Gigi"])
        
        with tab_umum:
            st.info("Akses publik untuk mengecek indikasi kesehatan gigi secara mandiri.")
            if st.button("Masuk sebagai Masyarakat Umum", use_container_width=True):
                st.session_state.role = "Masyarakat Umum"
                st.session_state.logged_in = True
                st.rerun() 
                
        with tab_dokter:
            st.warning("Akses khusus tenaga medis klinis. Memerlukan otentikasi.")
            password_input = st.text_input("Masukkan Password", type="password")
            if st.button("Login Panel Dokter", use_container_width=True):
                if password_input == "dokter123":
                    st.session_state.role = "Dokter Gigi (Analisis Klinis)"
                    st.session_state.logged_in = True
                    st.rerun()
                elif password_input == "":
                    st.error("Password tidak boleh kosong!")
                else:
                    st.error("Password salah! Akses ditolak.")

# ==========================================
# 6. HALAMAN DASHBOARD UTAMA (JIKA SUDAH LOGIN)
# ==========================================
else:
    # --- PANEL SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3011/3011387.png", width=80)
        st.title("Profil Pengguna")
        st.success(f"Masuk sebagai:\n**{st.session_state.role}**")
        
        if st.button("🚪 Keluar (Logout)", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.rerun()
            
        st.markdown("---")
        st.title("Status Sistem")
        
        # Indikator status LLM (menggantikan input box)
        if API_KEY:
            st.info("🟢 AI Narrative Terhubung")
        else:
            st.warning("🟡 AI Narrative Offline")
            
        st.markdown("---")
        st.caption("Sistem Analisis dan Deteksi Penyakit Gigi Berbasis AI (YOLOv8 & ViT)")
        st.caption("© 2026 - Tugas Akhir")

    # --- KONTEN UTAMA ---
    st.title("🦷 Sistem Analisis Kesehatan Gigi Terpadu")

    layanan_ai = st.radio(
        "Pilih Layanan Analisis Gigi:",
        ["Deteksi Kalkulus (YOLOv8)", "Diagnosis Karies (ViT-Base)"],
        horizontal=True
    )

    st.markdown("---")
    role_saat_ini = st.session_state.role

    # ------------------------------------------
    # OPSI A: DETEKSI KALKULUS (YOLO + LLM)
    # ------------------------------------------
    if layanan_ai == "Deteksi Kalkulus (YOLOv8)":
        col_kiri, col_kanan = st.columns([1, 1.5])
        
        with col_kiri:
            st.subheader("1. Pengaturan & Input Citra")
            metode_input = st.radio("Pilih Metode Input:", ["Unggah File dari Perangkat", "Gunakan Kamera Webcam"], key="metode_yolo")
            
            image = None
            if metode_input == "Unggah File dari Perangkat":
                uploaded_file = st.file_uploader("Pilih foto gigi...", type=["jpg", "jpeg", "png"], key="yolo_upload")
                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption='Foto yang Diunggah', use_container_width=True)
                    
            elif metode_input == "Gunakan Kamera Webcam":
                camera_file = st.camera_input("Posisikan gigi Anda dengan jelas di depan kamera", key="yolo_cam")
                if camera_file is not None:
                    image = Image.open(camera_file).convert('RGB')
            
            if image is not None:
                analyze_btn_yolo = st.button("🔍 Analisis Kalkulus", type="primary", use_container_width=True)

        with col_kanan:
            st.subheader("2. Hasil Analisis AI")
            if image is not None and analyze_btn_yolo:
                with st.spinner("YOLOv8 sedang memetakan area kalkulus..."):
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    results = model_yolo(img_bgr, conf=0.25)
                    boxes = results[0].boxes
                    jumlah_kalkulus = len(boxes)
                    
                    res_plotted_bgr = results[0].plot()
                    res_plotted_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)
                    time.sleep(0.5)
                    
                st.markdown("#### 🎯 Visualisasi Deteksi (Bounding Box)")
                st.image(res_plotted_rgb, caption=f'Ditemukan {jumlah_kalkulus} area kalkulus', use_container_width=True)
                
                if jumlah_kalkulus > 0:
                    if role_saat_ini == "Dokter Gigi (Analisis Klinis)":
                        st.error(f"⚠️ Peringatan Klinis: Terdeteksi {jumlah_kalkulus} titik penumpukan kalkulus yang memerlukan tindakan scaling.")
                    else:
                        st.warning(f"⚠️ Terdeteksi adanya karang gigi pada gigi Anda. Segera jadwalkan kunjungan ke dokter gigi.")
                else:
                    st.success("✅ Gigi tampak bersih dari penumpukan karang gigi (kalkulus).")

                st.markdown("---")
                st.markdown("#### 🩺 Laporan & Rekomendasi (Gemini LLM)")
                
                if not API_KEY:
                    st.info("⚠️ Menampilkan database offline (Fallback).")
                    st.write(get_offline_diagnosis_kalkulus(jumlah_kalkulus))
                else:
                    with st.spinner("Menyusun narasi medis dengan AI Gemini..."):
                        try:
                            client = genai.Client(api_key=API_KEY)
                            available_models = [m.name for m in client.models.list()]
                            target_model = next((name.replace('models/', '') for name in available_models if 'flash' in name.lower()), 
                                                available_models[0].replace('models/', '') if available_models else None)
                            
                            if jumlah_kalkulus > 0:
                                if role_saat_ini == "Masyarakat Umum":
                                    prompt_yolo = f"""Sistem Object Detection YOLOv8 telah mendeteksi adanya {jumlah_kalkulus} titik penumpukan kalkulus (karang gigi) pada foto pengguna.
Sebagai asisten kesehatan gigi yang ramah, berikan penjelasan dengan bahasa awam. 
Kamu WAJIB memformat jawabanmu TEPAT seperti struktur di bawah ini:

**Deskripsi:** [Jelaskan bahaya karang gigi bagi kesehatan mulut]
**Karakteristik Visual:** [Sebutkan ciri-ciri visual karang gigi]
**Rekomendasi:** [Sarankan dengan KUAT agar pengguna segera menjadwalkan kunjungan ke Dokter Gigi]"""
                                else:
                                    prompt_yolo = f"""Sistem Object Detection YOLOv8 telah mendeteksi adanya {jumlah_kalkulus} titik penumpukan kalkulus pada foto pasien.
Sebagai asisten klinis untuk Dokter Gigi, berikan penjelasan terstruktur. 
Kamu WAJIB memformat jawabanmu TEPAT seperti struktur di bawah ini:

**Deskripsi:** [Jelaskan dampak {jumlah_kalkulus} penumpukan kalkulus terhadap jaringan periodontal]
**Karakteristik Visual:** [Sebutkan ciri-ciri visual klinis karang gigi]
**Rekomendasi:** [Berikan rekomendasi tindakan klinis seperti scaling (supra/subgingival)]"""
                            else:
                                if role_saat_ini == "Masyarakat Umum":
                                    prompt_yolo = f"""Sistem AI tidak mendeteksi adanya kalkulus (karang gigi) pada foto pengguna.
Kamu WAJIB memformat jawabanmu TEPAT seperti struktur di bawah ini:

**Deskripsi:** [Berikan pujian singkat dan jelaskan arti ketiadaan karang gigi]
**Karakteristik Visual:** [Sebutkan tampilan gigi dan gusi yang bersih]
**Rekomendasi:** [Edukasi cara menyikat gigi yang benar]"""
                                else:
                                    prompt_yolo = f"""Sistem AI tidak mendeteksi kalkulus pada foto pasien.
Kamu WAJIB memformat jawabanmu TEPAT seperti struktur di bawah ini:

**Deskripsi:** [Jelaskan bahwa pasien memiliki kontrol plak yang baik]
**Karakteristik Visual:** [Sebutkan margin gingiva yang tampak bersih klinis]
**Rekomendasi:** [Edukasi tindakan preventif standar kedokteran gigi]"""
                            
                            response = client.models.generate_content(model=target_model, contents=prompt_yolo)
                            st.write(response.text)
                            st.caption(f"✔️ Dianalisis menggunakan {target_model}")
                            
                        except Exception as e:
                            st.error(f"Gagal terhubung ke LLM. Detail: {str(e)}")
                            st.write(get_offline_diagnosis_kalkulus(jumlah_kalkulus))
                            
            elif image is None:
                st.info("👈 Silakan unggah atau ambil foto pada panel di sebelah kiri.")

    # ------------------------------------------
    # OPSI B: DIAGNOSIS KARIES (ViT + LLM)
    # ------------------------------------------
    elif layanan_ai == "Diagnosis Karies (ViT-Base)":
        if not vit_model_loaded:
            st.error("Gagal memuat model AI! Pastikan file `vit_karies_final_model (2).pth` berada di folder yang sama.")
        else:
            col_kiri, col_kanan = st.columns([1, 1.5])
        
            with col_kiri:
                st.subheader("1. Pengaturan & Input Citra")
                metode_input_karies = st.radio("Pilih Metode Input:", ["Unggah File dari Perangkat", "Gunakan Kamera Webcam"], key="metode_vit")
                
                image = None
                if metode_input_karies == "Unggah File dari Perangkat":
                    uploaded_file = st.file_uploader("Pilih file gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], key="vit_upload")
                    if uploaded_file is not None:
                        image = Image.open(uploaded_file).convert('RGB')
                        st.image(image, caption="Citra yang diunggah", use_container_width=True)
                        
                elif metode_input_karies == "Gunakan Kamera Webcam":
                    camera_file = st.camera_input("Posisikan gigi Anda dengan jelas di depan kamera", key="vit_cam")
                    if camera_file is not None:
                        image = Image.open(camera_file).convert('RGB')
                
                if image is not None:
                    analyze_btn_vit = st.button("🔍 Analisis Tingkat Karies", type="primary", use_container_width=True)
        
            with col_kanan:
                st.subheader("2. Hasil Analisis AI")
                
                if image is not None and analyze_btn_vit:
                    with st.spinner("Memproses citra dengan Vision Transformer..."):
                        img_tensor = vit_transform(image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = model_inference(img_tensor)
                            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                            confidence, preds = torch.max(probabilities, 0)
                            
                            result = class_names[preds.item()]
                            confidence_pct = confidence.item() * 100
                            normal_prob = probabilities[class_names.index('Normal')].item() * 100
                            severe_prob = probabilities[class_names.index('Severe')].item() * 100
        
                        time.sleep(0.5)
                        
                    st.markdown("#### 🎯 Hasil Klasifikasi (ViT-Base)")
                    
                    is_anomaly = False
                    if result == "Normal" and severe_prob > 20.0:
                        is_anomaly = True
                        st.error(f"⚠️ **PERINGATAN ANOMALI MEDIS DETECTED!**", icon="🚨")
                        st.warning(f"Model: **Normal** ({normal_prob:.1f}%), Sinyal Kerusakan: **Severe** ({severe_prob:.1f}%).")
                    else:
                        if role_saat_ini == "Dokter Gigi (Analisis Klinis)":
                            st.success(f"**Tingkat Karies Terdeteksi:** {result} (Confidence: {confidence_pct:.2f}%)")
                        else:
                            st.success(f"**Hasil Pengecekan:** Ditemukan indikasi gigi {result} (Keyakinan AI: {confidence_pct:.2f}%)")
        
                    st.markdown("**Detail Probabilitas Kelas:**")
                    for idx, name in enumerate(class_names):
                        prob_val = probabilities[idx].item()
                        st.progress(prob_val, text=f"{name}: {prob_val*100:.1f}%")
        
                    st.markdown("---")
                    st.markdown("#### 🩺 Laporan Diagnosis Lengkap")
                    
                    if not API_KEY:
                        st.info("⚠️ Menampilkan database offline (Fallback).")
                        st.write(get_offline_diagnosis_karies(result))
                    else:
                        with st.spinner("Menyusun narasi medis dengan AI Gemini..."):
                            try:
                                client = genai.Client(api_key=API_KEY)
                                available_models = [m.name for m in client.models.list()]
                                target_model = next((name.replace('models/', '') for name in available_models if 'flash' in name.lower()), 
                                                    available_models[0].replace('models/', '') if available_models else None)
                                
                                if is_anomaly:
                                    if role_saat_ini == "Masyarakat Umum":
                                        prompt_text = f"""Sistem AI mendeteksi anomali pada foto gigi pengguna. Terlihat sehat, tapi ada sinyal lubang dalam {severe_prob:.2f}%.
Sebagai asisten kesehatan, berikan peringatan ramah namun tegas. Format wajib:
**Deskripsi:** [Jelaskan lubang tersembunyi]
**Karakteristik Visual:** [Sebutkan bayangan aneh di foto]
**Rekomendasi:** [Instruksi KUAT ke dokter gigi untuk Rontgen]"""
                                    else:
                                        prompt_text = f"""Sistem AI mendeteksi anomali karies 'Severe' sebesar {severe_prob:.2f}% pada pasien.
Sebagai asisten klinis, berikan penjelasan terstruktur. Format wajib:
**Deskripsi:** [Jelaskan potensi hidden caries]
**Karakteristik Visual:** [Sebutkan anomali struktural]
**Rekomendasi:** [Rekomendasi pemeriksaan radiografi]"""
                                else:
                                    if role_saat_ini == "Masyarakat Umum":
                                        if result != "Normal":
                                            prompt_text = f"""Sistem AI mendeteksi masalah gigi tingkat: {result}.
Sebagai asisten kesehatan, jelaskan dengan bahasa awam. Format wajib:
**Deskripsi:** [Penjelasan karies tingkat {result}]
**Karakteristik Visual:** [Ciri-ciri yang terlihat]
**Rekomendasi:** [Instruksi SEGERA ke dokter gigi]"""
                                        else:
                                            prompt_text = f"""Sistem AI mendeteksi gigi dalam kondisi: {result}.
Format wajib:
**Deskripsi:** [Jelaskan gigi sehat]
**Karakteristik Visual:** [Sebutkan permukaan utuh]
**Rekomendasi:** [Tips jaga kebersihan dan cek rutin]"""
                                    else:
                                        prompt_text = f"""Sistem AI ViT mendeteksi kondisi karies tingkat: {result}.
Sebagai asisten klinis, berikan penjelasan medis. Format wajib:
**Deskripsi:** [Deskripsi patologis karies {result}]
**Karakteristik Visual:** [Karakteristik visual klinis]
**Rekomendasi:** [Rekomendasi tindakan medis klinis]"""
                                
                                response = client.models.generate_content(model=target_model, contents=prompt_text)
                                st.write(response.text)
                                st.caption(f"✔️ Dianalisis menggunakan {target_model}")
                                
                            except Exception as e:
                                st.error(f"Gagal terhubung ke LLM Google. Detail: {str(e)}")
                                st.write(get_offline_diagnosis_karies(result))
                                
                elif image is None:
                    st.info("👈 Silakan unggah atau ambil foto pada panel di sebelah kiri.")
