import io
import os
import re
import zipfile

import cv2
import numpy as np
import pytesseract
import streamlit as st

try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ---------------------------------------------------------------
# Instrumenten-Zuordnung (Schreibweise im Scan -> Ordnername)
# ---------------------------------------------------------------
INSTRUMENTE_MAP = {
    "fluegelhorn": "Fluegelhorn", "flügelhorn": "Fluegelhorn",
    "trompete": "Trompete", "klarinette": "Klarinette",
    "saxophon": "Saxophon", "saxofon": "Saxophon",
    "tenorhorn": "Tenorhorn", "bariton": "Bariton",
    "posaune": "Posaune", "tuba": "Tuba",
    "waldhorn": "Horn", "horn": "Horn",
    "flöte": "Floete", "floete": "Floete",
    "schlagzeug": "Schlagzeug", "perkussion": "Schlagzeug",
}
# längste Begriffe zuerst, damit z. B. "Flügelhorn" vor "Horn" geprüft wird
INSTR_ALT = "|".join(re.escape(k) for k in sorted(INSTRUMENTE_MAP, key=len, reverse=True))


# --- PARSER-START ---
def parse_header_text(ocr_text):
    """Liest Nummer, Instrument, Stimme, Tonart und Titel aus dem OCR-Text."""
    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
    full_text = " ".join(lines)

    data = {"nummer": None, "instrument": None, "stimme": None,
            "tonart": None, "titel": None}

    # 1. Nummer ("Nr. 12", "No 12")
    m = re.search(r"\b(?:Nr|No)\.?\s*(\d{1,3})\b", full_text, re.IGNORECASE)
    if m:
        data["nummer"] = m.group(1).zfill(3)

    # 2. Instrument
    m = re.search(rf"\b({INSTR_ALT})\b", full_text, re.IGNORECASE)
    if m:
        gefunden = m.group(1)
        data["instrument"] = INSTRUMENTE_MAP[gefunden.lower()]

        # 3. Stimme: nur direkt vor oder nach dem Instrument ("1. Trompete", "Trompete 2")
        v = (re.search(rf"\b([1-4])\s*\.?\s*{re.escape(gefunden)}\b", full_text, re.IGNORECASE)
             or re.search(rf"\b{re.escape(gefunden)}\s*([1-4])\b", full_text, re.IGNORECASE))
        if v:
            data["stimme"] = v.group(1)

    # 4. Tonart ("in B", "in Es")
    t = re.search(r"\bin\s+(Es|As|B|C|D|F|G)\b", full_text)
    if t:
        data["tonart"] = t.group(1)

    # 5. Titel: Zeilen mit Instrument überspringen, Nummer entfernen, längste Zeile nehmen
    kandidaten = []
    for line in lines:
        if re.search(rf"\b({INSTR_ALT})\b", line, re.IGNORECASE):
            continue
        cleaned = re.sub(r"\b(?:Nr|No)\.?\s*\d{1,3}\b", "", line, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^\w\s-]", "", cleaned).strip()
        if len(cleaned) > 2:
            kandidaten.append(cleaned)
    if kandidaten:
        data["titel"] = max(kandidaten, key=len)[:60].strip().replace(" ", "_")

    return data
# --- PARSER-ENDE ---


def lade_bild(datei_bytes, dateiname):
    """Wandelt PDF (erste Seite) oder Bild in ein OpenCV-Bild um."""
    ext = os.path.splitext(dateiname)[1].lower()
    if ext == ".pdf":
        if not PDF_SUPPORT:
            return None
        seiten = convert_from_bytes(datei_bytes, dpi=200, first_page=1, last_page=1)
        if not seiten:
            return None
        return cv2.cvtColor(np.array(seiten[0]), cv2.COLOR_RGB2BGR)
    arr = np.frombuffer(datei_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@st.cache_data(show_spinner=False)
def analysiere(datei_bytes, dateiname, kopf_prozent):
    """OCR der Kopfzeile. Ergebnis wird zwischengespeichert (kein erneutes OCR bei jedem Klick)."""
    try:
        img = lade_bild(datei_bytes, dateiname)
    except Exception:
        return None
    if img is None:
        return None

    hoehe = img.shape[0]
    kopf = img[0:int(hoehe * kopf_prozent / 100), :]
    grau = cv2.cvtColor(kopf, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grau, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ocr_text = pytesseract.image_to_string(thresh, lang="deu+eng", config="--oem 3 --psm 6")
    data = parse_header_text(ocr_text)

    ok, png = cv2.imencode(".png", kopf)
    return {"data": data, "ocr_text": ocr_text, "kopf_png": png.tobytes() if ok else None}


def eindeutiger_pfad(pfad, benutzt):
    """Hängt _2, _3 ... an, falls der Pfad im ZIP schon existiert."""
    basis, ext = os.path.splitext(pfad)
    p, n = pfad, 1
    while p in benutzt:
        n += 1
        p = f"{basis}_{n}{ext}"
    benutzt.add(p)
    return p


# ---------------------------------------------------------------
# Oberfläche
# ---------------------------------------------------------------
st.set_page_config(page_title="Musikverein Noten-Sortierer", layout="wide")
st.title("🎵 Musikverein Noten-Sortierer")
st.write("Lade gescannte Notenblätter (PDF, JPG, PNG) hoch. Die App liest die Kopfzeile "
         "und benennt die Dateien automatisch um.")

with st.sidebar:
    st.header("Einstellungen")
    kopf_prozent = st.slider("Größe der Kopfzeile (% der Seitenhöhe)", 8, 40, 18)
    st.caption("Wird das Instrument oder der Titel abgeschnitten, Wert erhöhen.")

uploaded_files = st.file_uploader(
    "Notenblätter auswählen (Dateien oder Scans)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} Datei(en) ausgewählt.")

    zip_buffer = io.BytesIO()
    benutzte_pfade = set()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for uploaded_file in uploaded_files:
            datei_bytes = uploaded_file.getvalue()
            ext = os.path.splitext(uploaded_file.name)[1].lower()

            with st.spinner(f"Verarbeite {uploaded_file.name} ..."):
                ergebnis = analysiere(datei_bytes, uploaded_file.name, kopf_prozent)

            if ergebnis is None:
                st.warning(f"{uploaded_file.name} konnte nicht gelesen werden – "
                           f"liegt im Ordner _Unbekannt_Review.")
                pfad = eindeutiger_pfad(f"_Unbekannt_Review/{uploaded_file.name}", benutzte_pfade)
                zip_file.writestr(pfad, datei_bytes)
                st.divider()
                continue

            data = ergebnis["data"]
            ordner = data["instrument"] if data["instrument"] else "_Unbekannt_Review"

            teile = [
                data["nummer"],
                data["titel"],
                f"Stimme_{data['stimme']}" if data["stimme"] else None,
                f"in_{data['tonart']}" if data["tonart"] else None,
                data["instrument"],
            ]
            teile = [t for t in teile if t]
            neuer_name = ("_".join(teile) + ext) if teile else f"Unbekannt_{uploaded_file.name}"

            pfad = eindeutiger_pfad(f"{ordner}/{neuer_name}", benutzte_pfade)
            zip_file.writestr(pfad, datei_bytes)

            col1, col2 = st.columns([1, 2])
            with col1:
                if ergebnis["kopf_png"]:
                    st.image(ergebnis["kopf_png"], caption=f"Erkannte Kopfzeile – {uploaded_file.name}")
            with col2:
                st.success(f"**Zielpfad:** `{pfad}`")
                st.json(data)
                with st.expander("Roher OCR-Text anzeigen"):
                    st.text(ergebnis["ocr_text"])
            st.divider()

    st.download_button(
        label="📦 Alle sortierten Noten als ZIP herunterladen",
        data=zip_buffer.getvalue(),
        file_name="Notenarchiv_Sortiert.zip",
        mime="application/zip",
    )
