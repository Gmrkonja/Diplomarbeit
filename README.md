# YOLOv8 Echtzeit-Objekterkennung

Kurzes Beispielprojekt zur Erkennung von Objekten über die Webcam mit YOLOv8 (ultralytics).

Voraussetzungen
- Python 3.8+
- Virtualenv (empfohlen)
- Optional: NVIDIA GPU + passende torch-cuda Installation für Beschleunigung

Installation (Windows, PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Falls du eine CUDA-fähige Torch-Version brauchst, installiere `torch` entsprechend deiner CUDA-Version (siehe https://pytorch.org).

Benutzung

Aus der Projektmappe ausführen:

```powershell
venv\Scripts\python.exe object_detector.py
```

Optionen:
- `--source`: Kameraindex (z.B. `0`) oder Pfad zu Video-Datei
- `--model`: Pfad zum YOLOv8 Modell (Standard `yolov8n.pt`). Falls nicht vorhanden, lade das Modell manuell herunter oder ändere auf ein vorhandenes Modell.
- `--conf`: Konfidenz-Threshold (Default 0.35)
- `--iou`: IOU für NMS (Default 0.45)
- `--save`: Pfad, um das Ausgabevideo zu speichern (mp4)
- `--no-fps`: Deaktiviert FPS-Anzeige
- `--exclude`: Komma-getrennte Klassennamen, die nicht angezeigt werden sollen, z.B. `--exclude dog,cat`. Standardmäßig werden Personen (`person`) **automatisch ausgeschlossen**.

Beispiel mit Kamera und höherer Konfidenz:

```powershell
venv\Scripts\python.exe object_detector.py --conf 0.5
```

Hinweis: Live-Tests benötigen Zugang zu deiner Webcam. Wenn die Anwendung sofort beendet wird, prüfe:
- Wird die Kamera von einer anderen Anwendung belegt?
- Ist der Index `0` der richtige? Probiere `--source 1` oder `--source 2`.
- Hast du die Berechtigung zum Kamerazugriff (z. B. in Windows-Privatsphäre-Einstellungen)?

Wenn du keine GPU hast, läuft das Modell auf CPU (langsamer).

### Alternative Modelle

Das Skript lädt per Default `yolov8n.pt` (Nano) – ein sehr kleines Modell, das schnell, aber weniger genau ist. Du kannst andere vorgefertigte Gewichte verwenden:

| Dateiname | Beschreibung | Geschwindigkeit | Genauigkeit |
|-----------|--------------|-----------------|-------------|
| `yolov8n.pt` | Nano (Standard) | sehr schnell | niedrig |
| `yolov8s.pt` | Small | schnell | besser |
| `yolov8m.pt` | Medium | moderat | gut |
| `yolov8l.pt` | Large | langsamer | sehr gut |
| `yolov8x.pt` | eXtra large | am langsamsten | höchste |

Ein größeres Modell erreichst du einfach über `--model yolov8l.pt` oder indem du das Gewicht herunterlädst und den Pfad angibst. Für die CPU‑Ausführung solltest du eher `n`/`s`-Modelle wählen.

Darüber hinaus existieren völlig andere Architekturen:
* **EfficientDet**, **DETR**, **CenterNet**, **SSD/MobileNet** – können mit OpenCV DNN, TensorFlow oder PyTorch verwendet werden und liefern unterschiedliche Kompromisse zwischen Größe und Genauigkeit.
* **YOLOv5/v7/v9** – ältere/experimentelle Versionen mit ähnlicher API.
* **Segmentierungsmodelle** (z. B. `yolov8n-seg.pt`) für präzisere Masken statt Bounding‑Boxen.

Wenn du ein eigenes Modell trainieren willst (z. B. mit einem spezifischen Datensatz), kannst du die ultralytics‑API oder deren Web‑UI nutzen; die exportierten Gewichte lassen sich direkt mit dem oben gezeigten `--model`-Parameter laden.

---

