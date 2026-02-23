# Test Data

Questa cartella contiene file multimediali per i test end-to-end dell'applicazione.

## Struttura

```
test_data/
├── videos/      - Video di test per upscaler e lipsync
├── audio/       - File audio per test lipsync
├── images/      - Immagini di test per lipsync
└── outputs/     - Output temporanei dei test (auto-puliti)
```

## Come usare

1. **Per testare l'Upscaler**: 
   - Inserisci un video in `videos/`
   - Esegui: `pytest test/test_integration.py::TestUpscalerIntegration -v -s`

2. **Per testare il LipSync**:
   - Inserisci un'immagine/video in `videos/` o `images/`
   - Inserisci un file audio in `audio/`
   - Esegui: `pytest test/test_integration.py::TestLipSyncIntegration -v -s`

## File suggeriti

I test funzionano meglio con:
- Video brevi (5-30 secondi)
- Audio brevi (5-30 secondi) 
- Immagini con volti chiari (risoluzione 512x512 o superiore)
- Formati: MP4 (video), WAV/MP3 (audio), JPG/PNG (immagini)

## Note

- Questa cartella è inclusa nel `.gitignore` per evitare di versionare file pesanti
- I file di output vengono automaticamente puliti dopo i test
- Aggiungi qui i tuoi file multimediali personali per testare modifiche al codice
