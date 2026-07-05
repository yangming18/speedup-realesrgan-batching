# 👄 LipSync - Guida Completa

## 📋 Panoramica

Il tab **LipSync** permette di sincronizzare le labbra di immagini o video con file audio utilizzando modelli AI avanzati. Puoi animare foto statiche o sostituire i movimenti labiali in video esistenti.

## 🤖 Modelli Disponibili

L'applicazione supporta **5 modelli AI** con caratteristiche diverse:

### 1. Wav2Lip (⚡⚡⚡⚡⚡ | ⭐⭐⭐)
**Il più veloce - Ideale per preview e test rapidi**

- **Velocità**: Molto veloce (~1-2 minuti per 30 secondi)
- **Qualità**: Base, sufficiente per anteprime
- **Memoria**: Basso consumo (~2GB RAM)
- **Pro**: Velocissimo, stabile, funziona anche su CPU
- **Contro**: Qualità inferiore, possibili artefatti
- **Quando usarlo**: Preview rapide, test, video a bassa risoluzione

### 2. Wav2Lip GAN (⚡⚡⚡⚡ | ⭐⭐⭐⭐) ✅ CONSIGLIATO
**Miglior compromesso qualità/velocità**

- **Velocità**: Veloce (~2-4 minuti per 30 secondi)
- **Qualità**: Ottima, risultati naturali
- **Memoria**: Medio consumo (~4GB RAM)
- **Pro**: Qualità superiore, meno artefatti, relativamente veloce
- **Contro**: Richiede GPU/MPS per performance ottimali
- **Quando usarlo**: Produzione standard, la maggior parte dei progetti

### 3. SadTalker (⚡⚡ | ⭐⭐⭐⭐)
**Animazione completa con espressioni facciali**

- **Velocità**: Lento (~8-12 minuti)
- **Qualità**: Molto alta, espressioni realistiche
- **Memoria**: Alto consumo (~6GB RAM)
- **Pro**: Anima espressioni, movimenti naturali, risultati fotorealistici, mantiene immagine completa
- **Contro**: Molto lento, solo per immagini statiche, alto uso memoria
- **Quando usarlo**: Animazione di ritratti/foto con massimo realismo
- **Nota**: Configurato per mantenere l'intera immagine sorgente (modalità `full`) senza crop sul viso

### 4. Video-Retalking (⚡ | ⭐⭐⭐⭐⭐)
**Qualità professionale - Il migliore disponibile**

- **Velocità**: Molto lento (~15-25 minuti per 30 secondi)
- **Qualità**: Eccellente, qualità broadcast
- **Memoria**: Molto alto consumo (~8GB+ RAM)
- **Pro**: Massima qualità possibile, risultati professionali, preserva dettagli
- **Contro**: Lentissimo, richiede GPU potente, alto uso memoria
- **Quando usarlo**: Produzioni professionali, video finali ad alta qualità

### 5. LivePortrait (⚡⚡⚡ | ⭐⭐⭐⭐)
**Animazione viso + torso superiore - Movimenti corpo naturali**

- **Velocità**: Media (~5-8 minuti)
- **Qualità**: Molto alta, movimenti naturali
- **Memoria**: Medio-alto consumo (~4-6GB RAM)
- **Pro**: Anima viso e torso superiore, movimenti corpo realistici, qualità alta, mantiene scena completa
- **Contro**: Richiede GPU, medio uso memoria, solo immagini statiche
- **Quando usarlo**: Animazione ritratti con movimenti del corpo superiore
- **Nota**: Ideale quando serve animare anche le spalle e parte superiore del torso

### 6. The Gargantuas Hybrid LipSync (⚡⚡⚡ | ⭐⭐⭐⭐⭐) 🔥 NOVITÀ
**Pipeline ibrida proprietaria - Wav2Lip GAN + GFPGAN Face Enhancement**

- **Velocità**: Media (~4-8 minuti per 30 secondi)
- **Qualità**: Eccellente, massima qualità complessiva
- **Memoria**: Medio consumo (~5GB RAM)
- **Pro**: Lip sync perfetto (Wav2Lip GAN), Face restoration automatica (GFPGAN), Migliora qualità globale del viso, Preserva identità facciale, Qualità professionale
- **Contro**: Più lento (doppia pipeline), Richiede GPU per prestazioni ottimali
- **Quando usarlo**: Quando serve il miglior compromesso tra lip-sync perfetto e face quality. Ideale per produzioni dove la qualità del viso è importante quanto il lip-sync
- **Nota**: Pipeline proprietaria che combina due modelli AI in sequenza per risultati ottimali

## 🚀 Guida Rapida

### Prima Configurazione

1. **Installa dipendenze** (già fatto se hai seguito il setup):
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download modelli automatico**: Tutti i modelli vengono scaricati automaticamente al primo utilizzo (richiede connessione internet).

3. **⚠️ Video-Retalking**: Download automatico al primo utilizzo, con fallback manuale
   
   Quando usi **Video-Retalking** per la prima volta:
   
   - 🔄 Il sistema **tenta** il download automatico (~2GB, 5-15 min)
   - ⚠️ Se fallisce (limitazioni Google Drive), usa il **download manuale dal browser**:
     1. Vai su: https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0
     2. Seleziona tutti i file → Download
     3. Metti i file in: `models/lipsync/video_retalking/checkpoints/`
   - 💾 Usi successivi: Immediato (modelli già presenti)
   - 📖 **Guida dettagliata**: `models/lipsync/video_retalking/INSTALL_MODELS.md`
   
   **Altri modelli** (Wav2Lip, SadTalker, LivePortrait) sono più leggeri e si scaricano automaticamente senza problemi.

### Come Usare

#### Per Immagini (Foto Parlanti)

1. **Carica immagine**: Upload di una foto con un volto chiaro
2. **Carica audio**: Upload del file audio (voce, musica, qualsiasi suono)
3. **Scegli modello**:
   - `wav2lip_gan`: Consigliato (veloce + qualità)
   - `sadtalker`: Per massimo realismo
4. **Clicca** "🚀 Genera Lip Sync"
5. **Scarica** il video risultante

**Risultato**: La foto si anima con labbra sincronizzate all'audio!

#### Per Video (Sostituisci Audio/Labbra)

1. **Carica video**: Upload del video con il volto
2. **Carica audio**: Il nuovo audio da sincronizzare
3. **Scegli modello**:
   - `wav2lip_gan`: Standard (buon equilibrio)
   - `video_retalking`: Produzione professionale
4. **Clicca** "🚀 Genera Lip Sync"
5. **Scarica** il video con nuovo lip-sync

**Risultato**: Video con labbra sincronizzate al nuovo audio!

## ⚙️ Impostazioni Avanzate

### Device (Dispositivo di Calcolo)
- **MPS (Apple Silicon)**: Consigliato per Mac M1/M2/M3
- **GPU (CUDA)**: Per schede NVIDIA
- **CPU**: Lento ma universale

### Resize Factor (Fattore di Ridimensionamento)
- **1**: Dimensione originale (consigliato)
- **2-4**: Riduce dimensione per velocizzare (qualità inferiore)
- **Usa 2-4** solo per video molto grandi (>1080p) o hardware limitato

### No Smooth (Disabilita Smoothing)
- **OFF (default)**: Applica smoothing temporale per transizioni fluide
- **ON**: Disabilita smoothing, può migliorare nitidezza in alcuni casi
- **Prova entrambi** se i risultati non ti soddisfano

### Batch Size Wav2Lip / Wav2Lip GAN
- **Face Detection Batch Size**: numero di frame usati per batch durante il rilevamento volto.
- **Wav2Lip GAN Batch Size**: numero di sample usati per batch durante l'inferenza GAN.
- **Google Colab A100 consigliato**: Face Detection Batch Size `64`, Wav2Lip GAN Batch Size `512`.
- Se usi GPU con meno VRAM, riduci prima `Wav2Lip GAN Batch Size`, poi `Face Detection Batch Size`.

## 📊 Tempi di Processing (Stimati)

**Hardware**: Apple M1/M2 (MPS) o GPU NVIDIA equivalente

| Modello | 10 secondi | 30 secondi | 1 minuto |
|---------|------------|------------|----------|
| Wav2Lip | ~30 sec | ~1 min | ~2 min |
| Wav2Lip GAN | ~1 min | ~2-3 min | ~4-6 min |
| SadTalker | ~3 min | ~8-10 min | ~15-20 min |
| Video-Retalking | ~5 min | ~15-20 min | ~30-40 min |
| LivePortrait | ~2 min | ~5-7 min | ~10-15 min |
| The Gargantuas Hybrid | ~1.5 min | ~4-6 min | ~8-12 min |

**Note**: 
- CPU è circa 3-5x più lento
- Prima esecuzione richiede download modelli
- Video ad alta risoluzione richiedono più tempo

## 🎯 Consigli per Risultati Ottimali

### ✅ Qualità Input

**Immagine/Video:**
- Volto ben illuminato e chiaro
- Preferibilmente frontale (± 30° tollerati)
- Risoluzione minima: 480p | Ottimale: 720p-1080p
- Un solo volto per frame (multipli potrebbero causare problemi)

**Audio:**
- Pulito, senza rumori di fondo eccessivi
- Volume medio (non troppo basso/alto)
- Formati supportati: MP3, WAV, AAC, M4A, OGG

### 🔄 Workflow Consigliato

1. **Test rapido**: Usa `wav2lip` con clip brevi (5-10 sec)
2. **Valuta risultato**: Se soddisfatto, procedi
3. **Produzione**: Usa `wav2lip_gan` per la versione finale
4. **Qualità massima**: Usa `video_retalking` solo se necessario

### ⚠️ Limitazioni

- **Volti**: Deve essere visibile e chiaro in ogni frame
- **Angolazioni**: Profili completi (>70°) non funzionano bene
- **Occlusioni**: Mani/oggetti davanti alla bocca causano problemi
- **Lunghezza**: Video >5 minuti richiedono molto tempo e memoria

## 🐛 Risoluzione Problemi

### "❌ Errore durante il processing"

**Controlla i dettagli** nella sezione "Info" sotto il video output. Gli errori comuni:

1. **No module named 'librosa'**: 
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **OpenCV error**: Già gestito dalla patch automatica. Se persiste:
   ```bash
   pip uninstall opencv-python opencv-python-headless
   pip install opencv-python==4.9.0.80
   ```
   Vedi [OPENCV_PATCH.md](OPENCV_PATCH.md) per dettagli

3. **SadTalker NumPy error** (`np.float` deprecated): 
   - **Patch automatico**: Viene applicato automaticamente all'avvio
   - Il sistema rileva e corregge automaticamente incompatibilità NumPy 2.0+
   - Se vedi errori come `'module numpy has no attribute float'`:
     * Riavvia l'applicazione (il patch viene applicato al primo avvio)
     * Verifica che `safetensors` sia installato: `pip install safetensors`
   - ⚠️ **Non modificare manualmente la versione di NumPy** (potrebbe rompere gli upscaler)

4. **File temp non creato**: 
   - Verifica spazio disco (almeno 2GB liberi)
   - Controlla permessi cartella `temp/`

5. **Out of memory**:
   - Chiudi altre applicazioni
   - Usa modello più leggero (wav2lip invece di video_retalking)
   - Aumenta resize_factor (2-4)
   - Processa video più corti

### "No face detected"

- Assicurati il volto sia ben visibile e illuminato
- Prova con angolazione più frontale
- Verifica che l'immagine non sia troppo piccola (<256px)

### Processing molto lento

- **Verifica device**: Deve essere MPS o GPU, non CPU
- **Prima esecuzione**: Download e inizializzazione modelli è lenta
- **Modello**: SadTalker e Video-Retalking sono intrinsecamente lenti
- **Soluzione**: Usa `wav2lip_gan` per velocità accettabile

### ffmpeg non trovato

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

**Windows:**
- Scarica da [ffmpeg.org](https://ffmpeg.org/download.html)
- Aggiungi al PATH di sistema

## 📦 Struttura File

```
Video Editor/
├── docs/
│   └── LIPSYNC.md          # Questa guida
├── models/
│   └── lipsync/            # Modelli auto-scaricati
│       ├── wav2lip/
│       │   └── checkpoints/
│       │       ├── wav2lip.pth          (~55MB)
│       │       └── wav2lip_gan.pth      (~55MB)
│       ├── sadtalker/      # Scaricato al primo uso (~2GB)
│       └── video_retalking/ # Scaricato al primo uso (~1.5GB)
├── tabs/
│   └── lipsync_tab.py      # Implementazione tab
└── temp/                   # File temporanei (auto-pulizia)
```

## 🔬 Dettagli Tecnici

### Pipeline di Processing

1. **Caricamento**: Input validation e preparazione
2. **Download modello**: Se necessario (solo prima volta)
3. **Face detection**: Rilevamento automatico volto
4. **Audio extraction**: Estrazione features audio
5. **Lip generation**: AI genera movimenti labiali sincronizzati
6. **Video encoding**: Composizione finale con audio (H.264 + AAC 192k)

### Requisiti Hardware

| Componente | Minimo | Consigliato | Ottimale |
|------------|--------|-------------|----------|
| RAM | 8GB | 16GB | 32GB+ |
| GPU/MPS | - | 4GB VRAM | 8GB+ VRAM |
| Disco | 5GB | 10GB | 20GB+ |
| CPU | 4 core | 8 core | 12+ core |

### Modelli - Repository Originali

- **Wav2Lip**: [Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- **SadTalker**: [OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker)
- **Video-Retalking**: [OpenTalker/video-retalking](https://github.com/OpenTalker/video-retalking)
- **LivePortrait**: [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait)

## 📚 Risorse Aggiuntive

- [README Principale](../README.md) - Panoramica completa applicazione
- [OpenCV Patch](OPENCV_PATCH.md) - Fix compatibilità OpenCV
- [Quick Start](QUICKSTART.md) - Guida rapida setup
- [Paper Wav2Lip](https://arxiv.org/abs/2008.10010)
- [Paper SadTalker](https://arxiv.org/abs/2211.12194)

## 🆘 Supporto

In caso di problemi:

1. ✅ Leggi questa guida completa
2. ✅ Controlla sezione "Risoluzione Problemi"
3. ✅ Verifica log dettagliati nella sezione "Info"
4. ✅ Controlla che tutte le dipendenze siano installate
5. ✅ Riavvia l'applicazione

---

**Versione**: 1.0 | **Ultimo aggiornamento**: Febbraio 2026
