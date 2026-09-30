import hashlib
import io
import os
import re
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from pypdf import PdfReader, PdfWriter

try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ---------------------------------------------------------------
# Instrumenten-Zuordnung (Schreibweise im Scan -> Ordnername)
# Neue Instrumente einfach hier ergänzen.
# ---------------------------------------------------------------
INSTRUMENTE_MAP = {
    "fluegelhorn": "Fluegelhorn", "flügelhorn": "Fluegelhorn",
    "trompete": "Trompete", "klarinette": "Klarinette",
    "altsaxophon": "Altsaxophon", "altsaxofon": "Altsaxophon",
    "tenorsaxophon": "Tenorsaxophon", "tenorsaxofon": "Tenorsaxophon",
    "baritonsaxophon": "Baritonsaxophon",
    "saxophon": "Saxophon", "saxofon": "Saxophon",
    "tenorhorn": "Tenorhorn", "bariton": "Bariton",
    "bassposaune": "Bassposaune", "posaune": "Posaune", "tuba": "Tuba",
    "waldhorn": "Horn", "horn": "Horn",
    "flöte": "Floete", "floete": "Floete", "piccolo": "Piccolo",
    "oboe": "Oboe", "fagott": "Fagott",
    "schlagzeug": "Schlagzeug", "perkussion": "Schlagzeug",
}
# längste Begriffe zuerst, damit z. B. "Flügelhorn" vor "Horn" geprüft wird
INSTR_ALT = "|".join(re.escape(k) for k in sorted(INSTRUMENTE_MAP, key=len, reverse=True))
INSTRUMENT_OPTIONEN = sorted(set(INSTRUMENTE_MAP.values()) | {"Partitur", "Direktion"})
STIMMEN_OPTIONEN = ["1", "2", "3", "4"]
TONARTEN = ["B", "Es", "C", "F", "As", "D", "G"]


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


def leere_daten():
    return {"nummer": None, "instrument": None, "stimme": None, "tonart": None, "titel": None}


def kopf_analysieren(img, kopf_prozent):
    """OCR der Kopfzeile eines Blatts. Gibt (Daten, OCR-Text, Vorschau-PNG) zurück."""
    hoehe = img.shape[0]
    kopf = img[0:int(hoehe * kopf_prozent / 100), :]
    grau = cv2.cvtColor(kopf, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grau, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ocr_text = pytesseract.image_to_string(thresh, lang="deu+eng", config="--oem 3 --psm 6")
    data = parse_header_text(ocr_text)

    breite = kopf.shape[1]
    if breite > 700:
        faktor = 700 / breite
        kopf = cv2.resize(kopf, None, fx=faktor, fy=faktor, interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", kopf)
    return data, ocr_text, (png.tobytes() if ok else None)


@st.cache_data(show_spinner=False, max_entries=20)
def analysiere_datei(datei_bytes, dateiname, kopf_prozent):
    """Geht jede Seite einer Datei einzeln durch. Ergebnis wird zwischengespeichert."""
    stem, ext = os.path.splitext(dateiname)
    ext = ext.lower()
    ergebnis = []

    if ext == ".pdf":
        if not PDF_SUPPORT:
            return []
        reader = PdfReader(io.BytesIO(datei_bytes))
        for i in range(len(reader.pages)):
            data, ocr_text, png = leere_daten(), "", None
            try:
                bilder = convert_from_bytes(datei_bytes, dpi=200, first_page=i + 1, last_page=i + 1)
                if bilder:
                    img = cv2.cvtColor(np.array(bilder[0]), cv2.COLOR_RGB2BGR)
                    data, ocr_text, png = kopf_analysieren(img, kopf_prozent)
            except Exception:
                pass

            # Diese eine Seite als eigenes PDF herauslösen
            writer = PdfWriter()
            writer.add_page(reader.pages[i])
            buf = io.BytesIO()
            writer.write(buf)

            ergebnis.append({
                "quelle": f"{dateiname} – Seite {i + 1}",
                "basis": f"{stem}_S{i + 1}",
                "ext": ".pdf",
                "bytes": buf.getvalue(),
                "data": data, "ocr": ocr_text, "png": png,
            })
    else:
        data, ocr_text, png = leere_daten(), "", None
        arr = np.frombuffer(datei_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            data, ocr_text, png = kopf_analysieren(img, kopf_prozent)
        ergebnis.append({
            "quelle": dateiname, "basis": stem, "ext": ext,
            "bytes": datei_bytes, "data": data, "ocr": ocr_text, "png": png,
        })
    return ergebnis


def wert(x):
    """Macht aus leeren Zellen/NaN ein None, sonst einen bereinigten Text."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    x = str(x).strip()
    return x or None


def eindeutiger_pfad(pfad, benutzt):
    """Hängt _2, _3 ... an, falls der Pfad schon vergeben ist."""
    basis, ext = os.path.splitext(pfad)
    p, n = pfad, 1
    while p in benutzt:
        n += 1
        p = f"{basis}_{n}{ext}"
    benutzt.add(p)
    return p


def baue_eintraege(tabelle, seiten):
    """Erzeugt aus der (ggf. korrigierten) Tabelle Ordner, Dateinamen und Inhalte."""
    benutzt = set()
    ergebnis = []
    for i, (_, row) in enumerate(tabelle.iterrows()):
        s = seiten[i]
        instr = wert(row["Instrument"])
        stimme = wert(row["Stimme"])
        tonart = wert(row["Tonart"])
        nummer = wert(row["Nummer"])
        titel = wert(row["Titel"])
        if nummer and nummer.isdigit():
            nummer = nummer.zfill(3)
        titel_datei = re.sub(r"[^\w-]+", "_", titel).strip("_") if titel else None

        if instr:
            ordner = f"{instr}/Stimme_{stimme}" if stimme else f"{instr}/Ohne_Stimme"
            teile = [nummer, titel_datei,
                     f"{instr}_{stimme}" if stimme else instr,
                     f"in_{tonart}" if tonart else None]
            name = "_".join(t for t in teile if t) + s["ext"]
        else:
            ordner = "_Unbekannt_Review"
            name = re.sub(r"[^\w.-]+", "_", s["basis"]) + s["ext"]

        pfad = eindeutiger_pfad(f"{ordner}/{name}", benutzt)
        ergebnis.append({
            "pfad": pfad, "dateiname": os.path.basename(pfad),
            "instrument": instr, "stimme": stimme, "nummer": nummer,
            "titel": titel, "tonart": tonart,
            "quelle": s["quelle"], "bytes": s["bytes"],
        })
    ergebnis.sort(key=lambda e: (e["instrument"] or "~", e["stimme"] or "~",
                                 e["nummer"] or "~", e["titel"] or ""))
    return ergebnis


def erstelle_zip(paare):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for pfad, daten in paare:
            z.writestr(pfad, daten)
    return buf.getvalue()


# ---------------------------------------------------------------
# Oberfläche
# ---------------------------------------------------------------
st.set_page_config(page_title="Musikverein Noten-Sortierer", layout="wide")
st.title("🎵 Musikverein Noten-Sortierer")
st.write("Lade gescannte Notenblätter (PDF, JPG, PNG) hoch. Jede Seite wird einzeln gelesen, "
         "nach Instrument und Stimme sortiert und kann danach gezielt heruntergeladen werden.")

with st.sidebar:
    st.header("Einstellungen")
    kopf_prozent = st.slider("Größe der Kopfzeile (% der Seitenhöhe)", 8, 40, 18)
    st.caption("Wird das Instrument oder der Titel abgeschnitten, Wert erhöhen.")

uploaded_files = st.file_uploader(
    "Notenblätter auswählen (Dateien oder Scans)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Noch keine Dateien hochgeladen.")
    st.stop()

if not PDF_SUPPORT:
    st.warning("PDF-Unterstützung fehlt (pdf2image/poppler). Nur Bilder werden verarbeitet.")

# --- Alle Seiten aller Dateien einzeln lesen ---
seiten = []
fortschritt = st.progress(0.0, text="Lese Notenblätter ...")
for k, f in enumerate(uploaded_files):
    fortschritt.progress(k / len(uploaded_files), text=f"Verarbeite {f.name} ...")
    seiten.extend(analysiere_datei(f.getvalue(), f.name, kopf_prozent))
fortschritt.empty()

if not seiten:
    st.error("Es konnten keine Seiten gelesen werden.")
    st.stop()

# --- Tabelle für Prüfung/Korrektur ---
zeilen = []
for s in seiten:
    d = s["data"]
    zeilen.append({
        "Quelle": s["quelle"],
        "Nummer": d["nummer"],
        "Titel": (d["titel"] or "").replace("_", " ") or None,
        "Instrument": d["instrument"],
        "Stimme": d["stimme"],
        "Tonart": d["tonart"],
    })
tabelle = pd.DataFrame(zeilen)
editor_key = "editor_" + hashlib.md5(
    ("".join(f"{s['quelle']}{len(s['bytes'])}" for s in seiten) + str(kopf_prozent)).encode()
).hexdigest()

tab_pruefen, tab_download = st.tabs(["1️⃣ Prüfen & korrigieren", "2️⃣ Auswählen & herunterladen"])

with tab_pruefen:
    st.write(f"**{len(seiten)} Blatt/Blätter** erkannt. Fehler in der Tabelle kannst du direkt "
             "korrigieren (Zelle anklicken). Die Änderungen gelten sofort für Ordner und Download.")
    bearbeitet = st.data_editor(
        tabelle,
        key=editor_key,
        hide_index=True,
        disabled=["Quelle"],
        column_config={
            "Nummer": st.column_config.TextColumn("Nr."),
            "Instrument": st.column_config.SelectboxColumn("Instrument", options=INSTRUMENT_OPTIONEN),
            "Stimme": st.column_config.SelectboxColumn("Stimme", options=STIMMEN_OPTIONEN),
            "Tonart": st.column_config.SelectboxColumn("Tonart", options=TONARTEN),
        },
    )

    ohne_instrument = [i for i in range(len(bearbeitet)) if wert(bearbeitet.iloc[i]["Instrument"]) is None]
    if ohne_instrument:
        with st.expander(f"⚠️ {len(ohne_instrument)} Blatt/Blätter ohne erkanntes Instrument – Kopfzeile ansehen"):
            for i in ohne_instrument[:40]:
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.caption(seiten[i]["quelle"])
                    if seiten[i]["png"]:
                        st.image(seiten[i]["png"])
                with c2:
                    st.text(seiten[i]["ocr"] or "(kein Text erkannt)")
            if len(ohne_instrument) > 40:
                st.caption(f"... und {len(ohne_instrument) - 40} weitere.")

eintraege = baue_eintraege(bearbeitet, seiten)

with tab_download:
    bekannt = [e for e in eintraege if e["instrument"]]
    unbekannt = [e for e in eintraege if not e["instrument"]]

    if not bekannt:
        st.warning("Noch kein Blatt hat ein Instrument. Im Tab „Prüfen & korrigieren“ Instrument wählen.")
    else:
        st.subheader("Übersicht")
        uebersicht = pd.crosstab(
            pd.Series([e["instrument"] for e in bekannt], name="Instrument"),
            pd.Series([e["stimme"] or "ohne" for e in bekannt], name="Stimme"),
        )
        st.dataframe(uebersicht)

        st.subheader("Stimme auswählen")
        instrumente = sorted({e["instrument"] for e in bekannt})
        c1, c2 = st.columns(2)
        with c1:
            instr_wahl = st.selectbox("Instrument", instrumente)
        stimmen = sorted({e["stimme"] or "ohne" for e in bekannt if e["instrument"] == instr_wahl})
        with c2:
            stimme_wahl = st.selectbox(
                "Stimme", ["alle"] + stimmen,
                format_func=lambda v: {"alle": "Alle Stimmen", "ohne": "Ohne Stimme"}.get(v, f"Stimme {v}"),
            )

        auswahl = [e for e in bekannt if e["instrument"] == instr_wahl
                   and (stimme_wahl == "alle" or (e["stimme"] or "ohne") == stimme_wahl)]
        st.write(f"**{len(auswahl)} Blatt/Blätter** ausgewählt")
        st.dataframe(
            pd.DataFrame([{"Nr.": e["nummer"], "Titel": e["titel"], "Stimme": e["stimme"],
                           "Tonart": e["tonart"], "Datei": e["dateiname"]} for e in auswahl]),
            hide_index=True,
        )

        flach = set()
        zip_auswahl = erstelle_zip([(eindeutiger_pfad(e["dateiname"], flach), e["bytes"]) for e in auswahl])
        zip_name = f"{instr_wahl}_{'alle_Stimmen' if stimme_wahl == 'alle' else 'Stimme_' + stimme_wahl}.zip"
        st.download_button(f"📦 {instr_wahl} – {'alle Stimmen' if stimme_wahl == 'alle' else 'Stimme ' + stimme_wahl} "
                           f"als ZIP herunterladen",
                           data=zip_auswahl, file_name=zip_name, mime="application/zip")

    st.divider()
    st.subheader("Komplettarchiv")
    st.caption("Alle Blätter in der Ordnerstruktur Instrument / Stimme. "
               "Nicht erkannte Blätter liegen in _Unbekannt_Review.")
    st.download_button("📦 Alles als ZIP herunterladen",
                       data=erstelle_zip([(e["pfad"], e["bytes"]) for e in eintraege]),
                       file_name="Notenarchiv_Sortiert.zip", mime="application/zip")
    if unbekannt:
        st.caption(f"{len(unbekannt)} Blatt/Blätter ohne Instrument sind im Ordner _Unbekannt_Review.")
