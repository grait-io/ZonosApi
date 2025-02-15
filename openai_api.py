# Abhängigkeiten (installieren mit: pip install -r requirements.txt)
# fastapi
# uvicorn
# pydantic
# torch
# torchaudio
# mamba-ssm
# huggingface_hub

from fastapi import FastAPI, Response, APIRouter
from io import BytesIO
from pydantic import BaseModel
from zonos.model import Zonos # Zonos importieren (benötigt zonos package)
from zonos.conditioning import make_cond_dict # Konditionierungsfunktionen importieren

app = FastAPI(title="My API", description="My awesome API", version="1.0.0")

class SpeechRequest(BaseModel):
    """
    Anfrage für die Sprachsynthese.
    """
    model: str
    """Das zu verwendende Sprachmodell."""
    voice: str
    """Die zu verwendende Stimme."""
    input: str
    """Der zu synthetisierende Eingabetext."""

MODEL_REPO_ID = "Zyphra/Zonos-v0.1-transformer" # Modell-Repo ID (Zyphra/Zonos-v0.1-transformer oder Zyphra/Zonos-v0.1-hybrid)
device = "cuda" if torch.cuda.is_available() else "cpu" # Device-Auswahl (CUDA falls verfügbar)
model = None # Modell-Variable

@app.on_event("startup") # Startup-Event-Handler für FastAPI
async def startup_event():
    global model
    print(f"Loading Zonos model: {MODEL_REPO_ID}...")
    model = Zonos.from_pretrained(MODEL_REPO_ID, device=device) # Zonos Modell laden
    model.requires_grad_(False).eval() # Modell in den Eval-Modus setzen
    print(f"Zonos model loaded successfully on device: {device}!")

@app.post("/v1/audio/speech", summary="Sprachsynthese erstellen")
async def create_speech(request: SpeechRequest):
    """
    Erstellt eine Sprachsynthese aus dem gegebenen Text.
    """
    global model # Zugriff auf globale Modell-Variable

    if model is None: # Sicherstellen, dass das Modell geladen ist
        return Response(status_code=500, content="Zonos model not loaded!") # Fehler zurückgeben, falls Modell nicht geladen

    input_text = request.input # Eingabetext aus der Anfrage extrahieren
    print(f"Generating speech for input text: '{input_text}'") # Log-Ausgabe

    # Konditionierungs-Dictionary erstellen (minimal Beispiel)
    cond_dict = make_cond_dict(
        text=input_text,
        language="en-us", # Standard-Sprache (kann später parametrisiert werden)
        device=device,
    )
    conditioning = model.prepare_conditioning(cond_dict) # Konditionierung vorbereiten

    # Audio-Codes generieren
    codes = model.generate(
        prefix_conditioning=conditioning,
        max_new_tokens=86 * 30, # Maximale Anzahl neuer Tokens (kann angepasst werden)
        cfg_scale=2.0, # CFG Scale (kann angepasst werden)
        batch_size=1,
        sampling_params=dict(min_p=0.15), # Sampling Parameter (kann angepasst werden)
    )

    # Audio-Codes dekodieren
    wav_out = model.autoencoder.decode(codes).cpu().detach() # Dekodieren und auf CPU verschieben
    sr_out = model.autoencoder.sampling_rate # Abtastrate extrahieren

    # WAV-Daten in BytesIO-Stream konvertieren (Dummy-WAV-Daten für den Moment)
    dummy_wav = BytesIO()
    # Hier müsste der wav_out Tensor in ein gültiges WAV-Format konvertiert und in dummy_wav geschrieben werden
    torchaudio.save(dummy_wav, wav_out, sr_out, format="wav") # Beispiel (ggf. anpassen)

    # Für den Moment geben wir einen einfachen Text zurück, da Audio-Tests hier nicht möglich sind
    # dummy_response = f"Text-to-Speech Verarbeitung für: '{input_text}' (Dummy-Antwort, Audio-Generierung nicht aktiv)"
    # return Response(content=dummy_response, media_type="text/plain") # Text-Antwort zurückgeben

    # Audio-Stream als Response zurückgeben (WAV-Format)
    return Response(content=dummy_wav.getvalue(), media_type="audio/wav") # WAV-Stream zurückgeben (Content-Type: audio/wav)