document.addEventListener('DOMContentLoaded', () => {
    // ===========================================
    // 1. LOGIKA DROPDOWN (AMAN)
    // ===========================================
    const dropdownToggle = document.querySelector('.dropdown-toggle');
    const dropdownMenu = document.querySelector('.dropdown');
    
    if (dropdownToggle && dropdownMenu) {
        dropdownToggle.addEventListener('click', (e) => {
            e.preventDefault(); 
            dropdownMenu.classList.toggle('active');
        });

        document.addEventListener('click', (e) => {
            if (!dropdownMenu.contains(e.target) && !dropdownToggle.contains(e.target)) {
                dropdownMenu.classList.remove('active');
            }
        });
    }

    // ===========================================
    // 2. LOGIKA KAMERA & PREDIKSI DENGAN MEDIAPIPE
    // ===========================================
    const startButton = document.getElementById('start-camera-btn');
    const videoElement = document.getElementById('webcam-video');
    const canvasElement = document.getElementById('canvas-frame');
    const resultElement = document.getElementById('prediction-result');
    const exerciseImg = document.querySelector('.exercise-img'); // Ambil elemen gambar latihan
    // --- BARIS BARU DITAMBAHKAN ---
    const videoWrapper = document.getElementById('video-wrapper'); // Ambil wrapper video/canvas
    // ------------------------------
    
    // Variabel global untuk menyimpan instance MediaPipe
    window.cameraInstance = null;
    window.poseInstance = null;
    
    // Periksa apakah semua elemen kamera diperlukan ada di DOM
    if (startButton && videoElement && canvasElement && resultElement && videoWrapper) { // Tambahkan videoWrapper ke cek
        const context = canvasElement.getContext('2d'); 

        // === EVENT LISTENER UNTUK TOMBOL KAMERA ===
        startButton.addEventListener('click', () => {
            const exerciseName = window.location.pathname.replace('/', '');
            
            // Cek jika kamera belum aktif (tidak ada instance kamera)
            if (!window.cameraInstance || !window.cameraInstance.stream || !window.cameraInstance.stream.active) {
                startWebcam(exerciseName);
            } else {
                stopWebcam();
            }
        });

        // === FUNGSI START WEBCAM (Menggunakan MediaPipe) ===
        function startWebcam(exerciseName) {
            // 1. TAMPILKAN WRAPPER DAN SEMBUNYIKAN GAMBAR STATIS
            if (exerciseImg) {
                exerciseImg.style.display = 'none';
            }
            // --- BARIS KRITIS DITAMBAHKAN ---
            videoWrapper.style.display = 'block'; // Tampilkan seluruh area kamera
            // --------------------------------
            
            // 2. Inisialisasi Model Pose MediaPipe
            window.poseInstance = new Pose({locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
            }});

            // Konfigurasi tambahan untuk Pose Model
            window.poseInstance.setOptions({
                modelComplexity: 1, // Default, bisa diubah ke 0 atau 2
                smoothLandmarks: true,
                enableSegmentation: false,
                smoothSegmentation: true,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            // 3. Konfigurasi hasil (Hasil PoseLandmarks akan dikirim ke Flask)
            window.poseInstance.onResults((results) => {
                // Tampilkan CANVAS untuk menggambar hasil
                canvasElement.style.display = 'block';

                context.save();
                context.clearRect(0, 0, canvasElement.width, canvasElement.height);
                
                // Gambar video ke canvas (sebagai latar belakang)
                context.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
                
                let keypointsToSend = [];
                if (results.poseLandmarks) {
                    // Gambar keypoint pose di atas canvas
                    drawConnectors(context, results.poseLandmarks, POSE_CONNECTIONS,
                                   {color: '#00FF00', lineWidth: 4});
                    drawLandmarks(context, results.poseLandmarks, 
                                  {color: '#FF0000', lineWidth: 2});
                    
                    // Format keypoint menjadi array sederhana yang bisa dibaca Flask
                    keypointsToSend = results.poseLandmarks.map(landmark => [
                        landmark.x, landmark.y, landmark.z, landmark.visibility
                    ]);
                }
                context.restore();

                // Kirim keypoint ke Flask untuk prediksi
                sendFrameForPrediction(exerciseName, keypointsToSend);
            });
            
            // 4. Inisialisasi Kamera dan mulai mengirim frame ke Pose
            // PENTING: Lebar dan Tinggi harus sama dengan yang di CSS/HTML (420x315)
            window.cameraInstance = new Camera(videoElement, {
                onFrame: async () => {
                    await window.poseInstance.send({image: videoElement});
                },
                width: 420, 
                height: 315
            });
            window.cameraInstance.start();

            startButton.innerHTML = '⏸ Stop Check';
            resultElement.innerHTML = 'Memulai deteksi pose...';
            
            // Set ukuran canvas segera (agar sesuai dengan Camera)
            canvasElement.width = 420;
            canvasElement.height = 315;
        }

        // === FUNGSI STOP WEBCAM ===
        function stopWebcam() {
            if (window.cameraInstance) {
                window.cameraInstance.stop(); 
                window.cameraInstance = null;
            }
            if (videoElement.srcObject) {
                const tracks = videoElement.srcObject.getTracks();
                tracks.forEach(track => track.stop());
                videoElement.srcObject = null;
            }
            
            videoElement.style.display = 'none';
            canvasElement.style.display = 'none'; // Sembunyikan Canvas
            
            // --- BARIS KRITIS DITAMBAHKAN ---
            videoWrapper.style.display = 'none'; // Sembunyikan seluruh area kamera
            // --------------------------------
            
            // Tampilkan kembali gambar latihan
            if (exerciseImg) {
                exerciseImg.style.display = 'block';
            }
            
            startButton.innerHTML = `<img src="{{ url_for('static', filename='images/Camera.png') }}" alt="Start Camera" class="camera-icon">`; 
            resultElement.innerHTML = ''; // Bersihkan hasil prediksi
        }

        // === FUNGSI PENGIRIMAN KEYPOINT KE FLASK ===
        // Menerima keypoints sebagai parameter dari MediaPipe
        function sendFrameForPrediction(exerciseName, keypoints) {
            
            // Check jika keypoint kosong (tidak terdeteksi)
            if (keypoints.length === 0) {
                 resultElement.innerHTML = `<p style="color: red; font-size: 1.2em; font-weight: 600;">INCORRECT (N/A%)</p><p>Gagal Mendeteksi Pose (Pastikan seluruh badan terlihat)</p>`;
                 return;
            }

            fetch('/predict_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    keypoints: keypoints, // MENGIRIM KEYPOINTS ASLI
                    exercise: exerciseName
                }),
            })
            .then(response => response.json())
            .then(data => {
                const color = data.prediction === 'CORRECT' ? 'lightgreen' : 'red';
                resultElement.innerHTML = `
                    <p style="color: ${color}; font-size: 1.2em; font-weight: 600;">
                        ${data.prediction} (${data.accuracy ? data.accuracy.toFixed(1) : 'N/A'}%)
                    </p>
                    <p>${data.feedback ? data.feedback : ''}</p>
                `;
            })
            .catch(error => {
                console.error('Error saat prediksi:', error);
                resultElement.innerHTML = `<span style="color: yellow;">⚠️ Error Komunikasi Server.</span>`;
            });
        }
    }
});