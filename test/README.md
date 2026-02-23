# Video Editor - Test Suite 🧪

Suite completa di test per l'applicazione Video Editor, che copre tutte le funzionalità principali: LipSync, Upscaler, Utils e Patch system.

## 📋 Indice

- [Installazione](#installazione)
- [Esecuzione Test](#esecuzione-test)
- [Struttura Test](#struttura-test)
- [Copertura Test](#copertura-test)
- [Best Practice](#best-practice)

## 🚀 Installazione

### 1. Installa dipendenze di test

```bash
# Dalla root del progetto
pip install -r requirements.txt
```

Questo installerà:
- `pytest>=7.4.0` - Framework di testing
- `pytest-cov>=4.1.0` - Coverage reporting
- `pytest-mock>=3.11.0` - Mock utilities

### 2. Verifica installazione

```bash
pytest --version
```

## 🧪 Esecuzione Test

### Esegui tutti i test

```bash
# Dalla root del progetto
pytest test/
```

### Esegui test specifici

```bash
# Test LipSync
pytest test/test_lipsync.py

# Test Upscaler
pytest test/test_upscaler.py

# Test Utils
pytest test/test_utils.py

# Test Patches
pytest test/test_patches.py

# Test Integration
pytest test/test_integration.py
```

### Esegui test con verbose output

```bash
pytest test/ -v
```

### Esegui test per un modulo specifico

```bash
# Solo test per SadTalker
pytest test/test_lipsync.py::TestLipSyncModels::test_model_quality_ratings -v

# Solo test per DeviceManager
pytest test/test_utils.py::TestDeviceManager -v
```

### Esegui test con coverage report

```bash
# Coverage report in terminale
pytest --cov=. test/

# Coverage report HTML
pytest --cov=. --cov-report=html test/

# Apri report HTML
open htmlcov/index.html
```

### Esegui test con markers

```bash
# Solo test rapidi (se definiti)
pytest -m "not slow" test/

# Solo test di integrazione
pytest test/test_integration.py
```

## 📁 Struttura Test

```
test/
├── __init__.py              # Package initialization
├── conftest.py              # Fixtures e configurazione pytest
├── test_lipsync.py          # Test per LipSync (tutti e 5 i modelli)
├── test_upscaler.py         # Test per Upscaler
├── test_utils.py            # Test per utils (DeviceManager, TempManager, I18n)
├── test_patches.py          # Test per patch NumPy (SadTalker, LivePortrait)
├── test_integration.py      # Test di integrazione end-to-end
└── README.md                # Questa documentazione
```

## 🎯 Copertura Test

### test_lipsync.py

✅ **Modelli testati**: wav2lip, wav2lip_gan, sadtalker, video_retalking, liveportrait

- `TestLipSyncModels`: Definizioni e metadata modelli
- `TestLipSyncProcessor`: Inizializzazione e gestione processor
- `TestLipSyncTab`: UI e funzionalità tab
- `TestLipSyncProcessing`: Metodi di processing
- `TestLipSyncErrorHandling`: Gestione errori
- `TestLipSyncI18n`: Traduzioni e internazionalizzazione
- `TestLipSyncPatches`: Applicazione patch NumPy

**Totale**: ~40 test

### test_upscaler.py

✅ **Funzionalità testate**: Image upscaling, video upscaling, model loading

- `TestUpscalerInitialization`: Inizializzazione tab
- `TestUpscalerModelLoading`: Caricamento modelli RealESRGAN
- `TestUpscalerImageProcessing`: Upscaling immagini
- `TestUpscalerVideoProcessing`: Upscaling video
- `TestUpscalerDeviceManagement`: Gestione dispositivi (CPU/CUDA/MPS)
- `TestUpscalerConfiguration`: Configurazione modelli
- `TestUpscalerErrorHandling`: Gestione errori
- `TestUpscalerFormats`: Formati input/output

**Totale**: ~25 test

### test_utils.py

✅ **Moduli testati**: DeviceManager, TempManager, I18n

- `TestDeviceManager`: Rilevamento e gestione dispositivi (CPU/CUDA/MPS)
- `TestTempManager`: Gestione file temporanei
- `TestI18n`: Sistema di traduzione EN/IT
- `TestOpenCVPatch`: Patch OpenCV (se implementato)
- `TestUtilsIntegration`: Integrazione tra utils

**Totale**: ~35 test

### test_patches.py

✅ **Patch testati**: SadTalker NumPy, LivePortrait NumPy

- `TestSadTalkerPatch`: Patch NumPy 2.0+ per SadTalker
- `TestLivePortraitPatch`: Patch NumPy 2.0+ per LivePortrait
- `TestPatchPatterns`: Pattern di sostituzione regex
- `TestPatchIntegration`: Integrazione e idempotenza
- `TestPatchLogging`: Logging e reporting
- `TestPatchEdgeCases`: Edge cases e error handling
- `TestPatchDocumentation`: Documentazione patch

**Totale**: ~30 test

### test_integration.py

✅ **Workflow testati**: Inizializzazione app, integrazione tabs, multi-language

- `TestApplicationInitialization`: Inizializzazione VideoEditorApp
- `TestTabsIntegration`: Integrazione tra tutti i tab
- `TestUtilsIntegration`: Integrazione utils con app
- `TestConfigIntegration`: Configurazione
- `TestThemeIntegration`: Custom theme
- `TestEndToEndWorkflow`: Workflow completi LipSync e Upscaler
- `TestErrorHandlingIntegration`: Gestione errori globale
- `TestPatchSystemIntegration`: Sistema patch integrato
- `TestLanguageSupport`: Supporto multi-lingua (EN/IT)
- `TestDependencies`: Dipendenze Python
- `TestSystemCompatibility`: Compatibilità sistema

**Totale**: ~50 test

### 📊 Riepilogo Totale

| File | Test | Copertura |
|------|------|-----------|
| test_lipsync.py | ~40 | LipSync completo (5 modelli) |
| test_upscaler.py | ~25 | Upscaler completo |
| test_utils.py | ~35 | Utils completi |
| test_patches.py | ~30 | Patch system completo |
| test_integration.py | ~50 | Integrazione end-to-end |
| **TOTALE** | **~180** | **Applicazione completa** |

## ✅ Best Practice

### 1. Esegui test prima di commit

```bash
# Esegui tutti i test
pytest test/

# Se passano, puoi fare commit
git add .
git commit -m "Your changes"
```

### 2. Verifica coverage regolarmente

```bash
# Genera coverage report
pytest --cov=. --cov-report=html test/

# Apri report
open htmlcov/index.html

# Target: >80% coverage
```

### 3. Test specifici durante sviluppo

```bash
# Stai modificando LipSync? Testa solo quello
pytest test/test_lipsync.py -v

# Stai aggiungendo un nuovo modello? Testa integrazione
pytest test/test_integration.py::TestEndToEndWorkflow -v
```

### 4. Mock per test veloci

I test usano mock per evitare:
- ❌ Download modelli AI (pesanti, >2GB)
- ❌ Processing video reale (lento)
- ❌ Operazioni GPU (non sempre disponibile)

✅ I test verificano la **logica**, non l'esecuzione reale.

### 5. Test fallisce? Debug con verbose

```bash
# Verbose output
pytest test/test_lipsync.py -v

# Ancora più verbose
pytest test/test_lipsync.py -vv

# Stop al primo fallimento
pytest test/ -x

# Print output anche per test passati
pytest test/ -s
```

## 🔧 Troubleshooting

### ImportError durante test

```bash
# Assicurati di essere nella root del progetto
cd /path/to/Video\ Editor

# Verifica che __pycache__ sia pulita
find . -name "__pycache__" -type d -exec rm -rf {} +

# Ri-esegui test
pytest test/
```

### Test falliscono per dipendenze mancanti

```bash
# Re-installa requirements
pip install -r requirements.txt

# Verifica installazione
pip list | grep pytest
```

### Coverage report non si genera

```bash
# Installa pytest-cov
pip install pytest-cov>=4.1.0

# Ri-esegui con coverage
pytest --cov=. --cov-report=html test/
```

## 🎓 Esempi d'Uso

### Scenario 1: Nuova funzionalità LipSync

```bash
# 1. Scrivi il codice (es. nuovo metodo in lipsync_tab.py)

# 2. Esegui test esistenti per verificare non hai rotto nulla
pytest test/test_lipsync.py -v

# 3. Aggiungi test per la nuova funzionalità in test_lipsync.py

# 4. Esegui test integrazione
pytest test/test_integration.py -v

# 5. Coverage check
pytest --cov=tabs/lipsync_tab.py test/test_lipsync.py
```

### Scenario 2: Fix bug in Upscaler

```bash
# 1. Riproduci il bug con un test
# Aggiungi test in test_upscaler.py che fallisce

# 2. Fixa il bug nel codice

# 3. Verifica il test ora passa
pytest test/test_upscaler.py::test_your_bugfix -v

# 4. Esegui tutti i test upscaler
pytest test/test_upscaler.py -v

# 5. Full test suite
pytest test/ -v
```

### Scenario 3: Refactoring DeviceManager

```bash
# 1. Esegui test baseline
pytest test/test_utils.py::TestDeviceManager -v

# 2. Fai refactoring

# 3. Ri-esegui test
pytest test/test_utils.py::TestDeviceManager -v

# 4. Test integrazione (device usato ovunque)
pytest test/test_integration.py -v

# 5. Full suite
pytest test/ -v
```

### Scenario 4: Pre-release check

```bash
# 1. Full test suite
pytest test/ -v

# 2. Coverage report
pytest --cov=. --cov-report=html test/

# 3. Verifica coverage > 80%
open htmlcov/index.html

# 4. Test specifici critici
pytest test/test_integration.py::TestEndToEndWorkflow -v

# 5. Se tutto passa, sei pronto per rilascio
```

## 📚 Risorse Aggiuntive

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Coverage](https://pytest-cov.readthedocs.io/)
- [Python Mock Objects](https://docs.python.org/3/library/unittest.mock.html)

## 🤝 Contribuire

Quando aggiungi nuove funzionalità:

1. ✍️ Scrivi test PRIMA del codice (TDD)
2. 🧪 Esegui test durante sviluppo
3. 📊 Verifica coverage > 80%
4. ✅ Tutti i test devono passare prima di merge
5. 📝 Documenta test complessi

## 📝 Note

- La cartella `test/` è in `.gitignore` - i test non vengono committati
- Coverage report (`htmlcov/`) è in `.gitignore`
- Test usano mock per velocità - non scaricano modelli AI
- Test sono indipendenti - ordine esecuzione non importa
- Fixtures in `conftest.py` sono condivise tra tutti i test

---

**Happy Testing! 🚀**
