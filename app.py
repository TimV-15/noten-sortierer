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
def parse_header_text(ocr_text, segmente=None):
    """Liest Nummer, Instrument, Stimme, Tonart und Titel aus dem OCR-Text der Kopfzeile."""
    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
    if segmente:                                     # zweiter OCR-Durchgang als zusätzliche Textquelle
        lines = lines + [sg["text"] for sg in segmente]
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

    # 5. Titel (aus den gefilterten Textstücken der Kopfzeile)
    data["titel"] = finde_titel(lines, segmente)

    return data


def zeilen_segmente(woerter, min_conf=20):
    """Gruppiert OCR-Wörter (mit Position) zu Textstücken: gleiche Zeile, ohne große Lücken."""
    w = [x for x in woerter if x["text"].strip() and x["conf"] >= min_conf]
    if not w:
        return []
    w.sort(key=lambda x: x["top"] + x["h"] / 2)
    zeilen = []
    for x in w:
        cy = x["top"] + x["h"] / 2
        if zeilen and abs(cy - zeilen[-1]["cy"]) <= 0.6 * max(x["h"], zeilen[-1]["h"]):
            z = zeilen[-1]
            z["w"].append(x)
            n = len(z["w"])
            z["cy"] = (z["cy"] * (n - 1) + cy) / n
            z["h"] = max(z["h"], x["h"])
        else:
            zeilen.append({"w": [x], "cy": cy, "h": x["h"]})

    teile = []
    for z in zeilen:
        ws = sorted(z["w"], key=lambda x: x["left"])
        aktuell = [ws[0]]
        for prev, x in zip(ws, ws[1:]):
            luecke = x["left"] - (prev["left"] + prev["w"])
            if luecke > 2.0 * max(prev["h"], x["h"]):      # große Lücke = anderes Textstück (z. B. Komponist rechts)
                teile.append(aktuell)
                aktuell = []
            aktuell.append(x)
        teile.append(aktuell)

    ergebnis = []
    for seg in teile:
        hs = sorted(x["h"] for x in seg)
        ergebnis.append({
            "text": " ".join(x["text"] for x in seg), "woerter": seg,
            "conf": sum(x["conf"] for x in seg) / len(seg), "hoehe": hs[len(hs) // 2],
            "top": min(x["top"] for x in seg), "left": seg[0]["left"],
        })
    ergebnis.sort(key=lambda t: (t["top"], t["left"]))
    return ergebnis


def finde_titel(lines, segmente=None):
    """Sucht den Titel: das größte, sicher gelesene Textstück ohne Instrument. Rauschen wird herausgefiltert."""
    kandidaten = []
    if segmente:
        for sg in segmente:
            if re.search(rf"\b({INSTR_ALT})\b", sg["text"], re.IGNORECASE):
                continue
            gute = []                                   # (Text, Sicherheit, Höhe)
            for wd in sg["woerter"]:
                t = re.sub(r"[^\w-]", "", wd["text"]).strip("_")
                if not t:
                    continue
                buchstaben = sum(c.isalpha() for c in t)
                if buchstaben == 0:
                    if t.isdigit() and 2 <= len(t) <= 4 and wd["conf"] >= 60:
                        gute.append((t, wd["conf"], wd["h"]))
                    continue
                mindest = 60 if buchstaben < 4 else 45      # kurze Wörter müssen sicherer gelesen sein
                if wd["conf"] < mindest or buchstaben < 2 or buchstaben / len(t) < 0.75:
                    continue
                gute.append((t, wd["conf"], wd["h"]))
            if not gute:
                continue
            hmax = max(h for _, _, h in gute)
            gute = [g for g in gute if g[2] >= 0.45 * hmax]   # viel kleinere Zeichen = Rauschen
            text = re.sub(r"\b(?:Nr|No)\s*\d{1,3}\b", "", " ".join(g[0] for g in gute),
                          flags=re.IGNORECASE).strip()
            if sum(c.isalpha() for c in text) < 4:
                continue
            if not any(sum(c.isalpha() for c in t) >= 3 for t in text.split()):
                continue
            hoehen = sorted(g[2] for g in gute)
            hoehe = hoehen[len(hoehen) // 2]
            conf = sum(g[1] for g in gute) / len(gute)
            kandidaten.append((hoehe * conf / 100.0, text))
        if not kandidaten:
            return None
        text = max(kandidaten, key=lambda k: k[0])[1]
    else:
        for line in lines:
            if re.search(rf"\b({INSTR_ALT})\b", line, re.IGNORECASE):
                continue
            cleaned = re.sub(r"\b(?:Nr|No)\.?\s*\d{1,3}\b", "", line, flags=re.IGNORECASE)
            cleaned = re.sub(r"[^\w\s-]", "", cleaned).strip()
            if len(cleaned) > 2:
                kandidaten.append((len(cleaned), cleaned))
        if not kandidaten:
            return None
        text = max(kandidaten, key=lambda k: k[0])[1]

    if text.isupper():
        text = text.title()
    return text[:60].strip().replace(" ", "_")


def parse_footer_text(ocr_text):
    """Liest die Seitenzahl aus der Fußzeile: 'Seite 2 von 3', '2/3', 'S. 2', '- 2 -', '2'.
    Gibt (seite, von) zurück, jeweils None wenn nicht erkannt."""
    zeilen = [l.strip() for l in ocr_text.split("\n") if l.strip()]
    text = " ".join(zeilen)
    nr = von = None

    m = re.search(r"\b(?:Seite|Page|S\.?|p\.?)\s*(\d{1,2})\s*(?:von|of|/)\s*(\d{1,2})\b", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\b(\d{1,2})\s*(?:von|of|/)\s*(\d{1,2})\b", text, re.IGNORECASE)
    if m:
        nr, von = int(m.group(1)), int(m.group(2))
        if nr < 1 or nr > von or von > 20:      # unplausibel (z. B. Datum) -> verwerfen
            nr = von = None
    else:
        m = re.search(r"\b(?:Seite|Page|S\.|p\.)\s*(\d{1,2})\b", text, re.IGNORECASE)
        if m:
            nr = int(m.group(1))
        else:
            for z in reversed(zeilen):
                m = re.fullmatch(r"\W*(\d{1,2})\W*", z)
                if m:
                    nr = int(m.group(1))
                    break
        if nr is not None and not (1 <= nr <= 99):
            nr = None
    return nr, von
# --- PARSER-ENDE ---


def leere_daten():
    return {"nummer": None, "instrument": None, "stimme": None, "tonart": None, "titel": None}


def vorschau_png(bild):
    breite = bild.shape[1]
    if breite > 700:
        bild = cv2.resize(bild, None, fx=700 / breite, fy=700 / breite, interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", bild)
    return png.tobytes() if ok else None


def notenlinien_entfernen(bw):
    """Entfernt lange waagerechte Linien (Notenlinien, Unterstreichungen), damit sie keine Buchstabensuppe erzeugen."""
    invers = cv2.bitwise_not(bw)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, bw.shape[1] // 12), 1))
    linien = cv2.morphologyEx(invers, cv2.MORPH_OPEN, kern)
    return cv2.bitwise_not(cv2.subtract(invers, linien))


def ocr_woerter(bild):
    """OCR mit Position und Sicherheit (Konfidenz) pro Wort."""
    d = pytesseract.image_to_data(bild, lang="deu+eng", config="--oem 3 --psm 11",
                                  output_type=pytesseract.Output.DICT)
    woerter = []
    for i, t in enumerate(d["text"]):
        t = str(t).strip()
        if not t:
            continue
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < 0:
            continue
        woerter.append({"text": t, "conf": conf, "left": d["left"][i], "top": d["top"][i],
                        "w": d["width"][i], "h": d["height"][i]})
    return woerter


def kopf_analysieren(img, kopf_prozent):
    """OCR der Kopfzeile. Gibt (Daten, OCR-Text, Vorschau-PNG, Textstücke) zurück."""
    hoehe = img.shape[0]
    kopf = img[0:int(hoehe * kopf_prozent / 100), :]
    grau = cv2.cvtColor(kopf, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(grau, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = notenlinien_entfernen(bw)

    ocr_text = pytesseract.image_to_string(bw, lang="deu+eng", config="--oem 3 --psm 6")
    segmente = zeilen_segmente(ocr_woerter(bw))
    data = parse_header_text(ocr_text, segmente)
    info = [{"Text": sg["text"], "Sicherheit": round(sg["conf"]), "Höhe": sg["hoehe"]} for sg in segmente]
    return data, ocr_text, vorschau_png(kopf), info


def fuss_analysieren(img, fuss_prozent):
    """OCR der Fußzeile. Gibt (Seite, von, OCR-Text, Vorschau-PNG) zurück."""
    hoehe = img.shape[0]
    fuss = img[int(hoehe * (1 - fuss_prozent / 100)):, :]
    grau = cv2.cvtColor(fuss, cv2.COLOR_BGR2GRAY)
    grau = cv2.resize(grau, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)   # kleine Schrift vergrößern
    _, thresh = cv2.threshold(grau, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ocr_text = pytesseract.image_to_string(thresh, lang="deu+eng", config="--oem 3 --psm 6")
    nr, von = parse_footer_text(ocr_text)
    return nr, von, ocr_text, vorschau_png(fuss)


def seite_auswerten(img, kopf_prozent, fuss_prozent):
    data, kopf_ocr, kopf_png, segmente = kopf_analysieren(img, kopf_prozent)
    nr, von, fuss_ocr, fuss_png = fuss_analysieren(img, fuss_prozent)
    return {"data": data, "ocr": kopf_ocr, "png": kopf_png, "segmente": segmente,
            "fnr": nr, "fvon": von, "fuss_ocr": fuss_ocr, "fuss_png": fuss_png}


LEERE_SEITE = {"data": None, "ocr": "", "png": None, "segmente": [], "fnr": None, "fvon": None,
               "fuss_ocr": "", "fuss_png": None}


@st.cache_data(show_spinner=False, max_entries=20)
def analysiere_datei(datei_bytes, dateiname, kopf_prozent, fuss_prozent):
    """Geht jede Seite einer Datei einzeln durch. Ergebnis wird zwischengespeichert."""
    stem, ext = os.path.splitext(dateiname)
    ext = ext.lower()
    ergebnis = []

    if ext == ".pdf":
        if not PDF_SUPPORT:
            return []
        reader = PdfReader(io.BytesIO(datei_bytes))
        for i in range(len(reader.pages)):
            info = dict(LEERE_SEITE, data=leere_daten())
            try:
                bilder = convert_from_bytes(datei_bytes, dpi=200, first_page=i + 1, last_page=i + 1)
                if bilder:
                    img = cv2.cvtColor(np.array(bilder[0]), cv2.COLOR_RGB2BGR)
                    info = seite_auswerten(img, kopf_prozent, fuss_prozent)
            except Exception:
                pass

            # Diese eine Seite als eigenes PDF herauslösen
            writer = PdfWriter()
            writer.add_page(reader.pages[i])
            buf = io.BytesIO()
            writer.write(buf)

            ergebnis.append(dict(info, datei=dateiname,
                                 quelle=f"{dateiname} – Seite {i + 1}",
                                 basis=f"{stem}_S{i + 1}", ext=".pdf", bytes=buf.getvalue()))
    else:
        info = dict(LEERE_SEITE, data=leere_daten())
        img = cv2.imdecode(np.frombuffer(datei_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            info = seite_auswerten(img, kopf_prozent, fuss_prozent)
        ergebnis.append(dict(info, datei=dateiname, quelle=dateiname,
                             basis=stem, ext=ext, bytes=datei_bytes))
    return ergebnis


def erkenne_fortsetzungen(seiten, auch_ohne_fussnummer=True):
    """Bestimmt, welche Seiten die Fortsetzung des vorherigen Blatts sind (gleiche Datei)."""
    ergebnis = []
    for i, s in enumerate(seiten):
        d, nr = s["data"], s["fnr"]
        vorher = seiten[i - 1] if i > 0 and seiten[i - 1]["datei"] == s["datei"] else None
        fort = False
        if vorher is not None:
            if not d["instrument"]:
                # keine Kopfzeile erkannt: Fußzeile entscheidet (Seite 1 = neues Stück)
                fort = (nr > 1) if nr is not None else auch_ohne_fussnummer
            elif nr is not None and nr > 1:
                # Kopfzeile wiederholt sich auf Folgeseiten -> gleiches Stück, wenn alles gleich ist
                dv = vorher["data"]
                fort = (d["instrument"] == dv["instrument"] and d["stimme"] == dv["stimme"]
                        and d["nummer"] == dv["nummer"])
        ergebnis.append(fort)
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


def zahl(x):
    w = wert(x)
    try:
        return int(float(w)) if w is not None else None
    except ValueError:
        return None


def ist_wahr(x):
    try:
        return bool(x) and not pd.isna(x)
    except (TypeError, ValueError):
        return False


def eindeutiger_pfad(pfad, benutzt):
    """Hängt _2, _3 ... an, falls der Pfad schon vergeben ist."""
    basis, ext = os.path.splitext(pfad)
    p, n = pfad, 1
    while p in benutzt:
        n += 1
        p = f"{basis}_{n}{ext}"
    benutzt.add(p)
    return p


def fuege_pdfs_zusammen(liste_bytes):
    writer = PdfWriter()
    for b in liste_bytes:
        for seite in PdfReader(io.BytesIO(b)).pages:
            writer.add_page(seite)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def baue_eintraege(tabelle, seiten, zusammenfuegen=True):
    """Gruppiert Seiten zu Stücken und erzeugt Ordner, Dateinamen und Inhalte."""
    # 1. Seiten zu Stücken zusammenfassen (Fortsetzung = gehört zum Stück davor)
    gruppen = []
    for i, (_, row) in enumerate(tabelle.iterrows()):
        s = seiten[i]
        if ist_wahr(row["Fortsetzung"]) and gruppen and gruppen[-1]["datei"] == s["datei"]:
            gruppen[-1]["idx"].append(i)
        else:
            gruppen.append({"datei": s["datei"], "idx": [i], "row": row})

    benutzt = set()
    ergebnis = []
    for g in gruppen:
        row, idx = g["row"], g["idx"]
        instr = wert(row["Instrument"])
        stimme = wert(row["Stimme"])
        tonart = wert(row["Tonart"])
        nummer = wert(row["Nummer"])
        titel = wert(row["Titel"])
        if nummer and nummer.isdigit():
            nummer = nummer.zfill(3)
        titel_datei = re.sub(r"[^\w-]+", "_", titel).strip("_") if titel else None

        n = len(idx)
        soll_werte = [zahl(tabelle.iloc[i]["von"]) for i in idx]
        soll_werte = [v for v in soll_werte if v]
        soll = max(soll_werte) if soll_werte else None
        vollstaendig = None if soll is None else (n >= soll)

        alle_pdf = all(seiten[i]["ext"] == ".pdf" for i in idx)
        if zusammenfuegen and alle_pdf:
            einheiten = [(fuege_pdfs_zusammen([seiten[i]["bytes"] for i in idx]),
                          "", ".pdf", seiten[idx[0]]["basis"])]
        else:
            einheiten = [(seiten[i]["bytes"], f"_S{k + 1}" if n > 1 else "",
                          seiten[i]["ext"], seiten[i]["basis"]) for k, i in enumerate(idx)]

        for daten_bytes, suffix, ext, basis in einheiten:
            if instr:
                ordner = f"{instr}/Stimme_{stimme}" if stimme else f"{instr}/Ohne_Stimme"
                teile = [nummer, titel_datei,
                         f"{instr}_{stimme}" if stimme else instr,
                         f"in_{tonart}" if tonart else None]
                name = "_".join(t for t in teile if t) + suffix + ext
            else:
                ordner = "_Unbekannt_Review"
                name = re.sub(r"[^\w.-]+", "_", basis) + ext
            pfad = eindeutiger_pfad(f"{ordner}/{name}", benutzt)
            ergebnis.append({
                "pfad": pfad, "dateiname": os.path.basename(pfad),
                "instrument": instr, "stimme": stimme, "nummer": nummer,
                "titel": titel, "tonart": tonart,
                "quelle": seiten[idx[0]]["quelle"] + (f" (+{n - 1})" if n > 1 else ""),
                "bytes": daten_bytes,
                "seiten": n, "soll": soll, "vollstaendig": vollstaendig,
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


def seiten_text(e):
    text = str(e["seiten"])
    if e["soll"]:
        text += f" von {e['soll']}"
    if e["vollstaendig"] is False:
        text += " ⚠️ unvollständig"
    return text


# ---------------------------------------------------------------
# Oberfläche
# ---------------------------------------------------------------
st.set_page_config(page_title="Musikverein Noten-Sortierer", layout="wide")
st.title("🎵 Musikverein Noten-Sortierer")
st.write("Lade gescannte Notenblätter (PDF, JPG, PNG) hoch. Jede Seite wird einzeln gelesen (Kopf- und Fußzeile), "
         "nach Instrument und Stimme sortiert, mehrseitige Stücke werden zusammengeführt.")

with st.sidebar:
    st.header("Einstellungen")
    kopf_prozent = st.slider("Kopfzeile (% der Seitenhöhe)", 8, 40, 18)
    fuss_prozent = st.slider("Fußzeile (% der Seitenhöhe)", 4, 20, 8)
    st.caption("Wird etwas abgeschnitten, den jeweiligen Wert erhöhen.")
    auto_fortsetzung = st.checkbox("Seiten ohne erkannte Kopfzeile als Fortsetzung behandeln", value=True,
                                   help="Gilt, wenn die Fußzeile keine Seitenzahl liefert. "
                                        "Steht dort „Seite 1“, beginnt immer ein neues Stück.")
    zusammenfuegen = st.checkbox("Mehrseitige Stücke zu einer PDF zusammenfügen", value=True)

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
    seiten.extend(analysiere_datei(f.getvalue(), f.name, kopf_prozent, fuss_prozent))
fortschritt.empty()

if not seiten:
    st.error("Es konnten keine Seiten gelesen werden.")
    st.stop()

# --- Tabelle für Prüfung/Korrektur ---
fortsetzungen = erkenne_fortsetzungen(seiten, auto_fortsetzung)
tabelle = pd.DataFrame([{
    "Quelle": s["quelle"],
    "Fortsetzung": fortsetzungen[i],
    "Nummer": s["data"]["nummer"],
    "Titel": (s["data"]["titel"] or "").replace("_", " ") or None,
    "Instrument": s["data"]["instrument"],
    "Stimme": s["data"]["stimme"],
    "Tonart": s["data"]["tonart"],
    "Seite": s["fnr"],
    "von": s["fvon"],
} for i, s in enumerate(seiten)])
tabelle["Seite"] = pd.to_numeric(tabelle["Seite"], errors="coerce")
tabelle["von"] = pd.to_numeric(tabelle["von"], errors="coerce")

editor_key = "editor_" + hashlib.md5(
    ("".join(f"{s['quelle']}{len(s['bytes'])}" for s in seiten)
     + f"{kopf_prozent}-{fuss_prozent}-{auto_fortsetzung}").encode()
).hexdigest()

tab_pruefen, tab_download = st.tabs(["1️⃣ Prüfen & korrigieren", "2️⃣ Auswählen & herunterladen"])

with tab_pruefen:
    st.write(f"**{len(seiten)} Seite(n)** gelesen. Fehler kannst du direkt in der Tabelle korrigieren "
             "(Zelle anklicken).")
    st.caption("Ist bei **Fortsetzung** ein Haken gesetzt, gehört die Seite zum Stück davor und übernimmt dessen "
               "Nr., Titel, Instrument, Stimme und Tonart. Haken entfernen = neues Stück. "
               "Seite / von = aus der Fußzeile gelesen.")
    bearbeitet = st.data_editor(
        tabelle,
        key=editor_key,
        hide_index=True,
        disabled=["Quelle"],
        column_config={
            "Fortsetzung": st.column_config.CheckboxColumn("Fortsetzung"),
            "Nummer": st.column_config.TextColumn("Nr."),
            "Instrument": st.column_config.SelectboxColumn("Instrument", options=INSTRUMENT_OPTIONEN),
            "Stimme": st.column_config.SelectboxColumn("Stimme", options=STIMMEN_OPTIONEN),
            "Tonart": st.column_config.SelectboxColumn("Tonart", options=TONARTEN),
            "Seite": st.column_config.NumberColumn("Seite", min_value=1, max_value=99, step=1, format="%d"),
            "von": st.column_config.NumberColumn("von", min_value=1, max_value=20, step=1, format="%d"),
        },
    )

    eintraege = baue_eintraege(bearbeitet, seiten, zusammenfuegen)
    st.write(f"➡️ **{len(eintraege)} Datei(en)** aus {len(seiten)} Seite(n).")

    unvollstaendig = [e for e in eintraege if e["vollstaendig"] is False]
    if unvollstaendig:
        st.warning(f"{len(unvollstaendig)} Stück(e) sind laut Fußzeile unvollständig: "
                   + "; ".join(f"{e['dateiname']} ({e['seiten']} von {e['soll']} Seiten)"
                               for e in unvollstaendig[:10])
                   + (" ..." if len(unvollstaendig) > 10 else ""))

    ohne_instrument = [i for i in range(len(bearbeitet))
                       if wert(bearbeitet.iloc[i]["Instrument"]) is None
                       and not ist_wahr(bearbeitet.iloc[i]["Fortsetzung"])]
    if ohne_instrument:
        with st.expander(f"⚠️ {len(ohne_instrument)} Blatt/Blätter ohne erkanntes Instrument – Kopf- und Fußzeile ansehen"):
            for i in ohne_instrument[:40]:
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.caption(seiten[i]["quelle"])
                    if seiten[i]["png"]:
                        st.image(seiten[i]["png"])
                    if seiten[i]["fuss_png"]:
                        st.image(seiten[i]["fuss_png"])
                with c2:
                    st.text("Kopfzeile:\n" + (seiten[i]["ocr"] or "(kein Text erkannt)"))
                    st.text("Fußzeile:\n" + (seiten[i]["fuss_ocr"] or "(kein Text erkannt)"))
            if len(ohne_instrument) > 40:
                st.caption(f"... und {len(ohne_instrument) - 40} weitere.")

    with st.expander("🔍 Diagnose: Kopf- und Fußzeile einer Seite ansehen"):
        idx = st.selectbox("Seite", list(range(len(seiten))), format_func=lambda i: seiten[i]["quelle"])
        st.caption("Kopfzeile (Ausschnitt):")
        if seiten[idx]["png"]:
            st.image(seiten[idx]["png"])
        st.caption("Fußzeile (Ausschnitt):")
        if seiten[idx]["fuss_png"]:
            st.image(seiten[idx]["fuss_png"])
        st.caption("Gefundene Textstücke der Kopfzeile (Sicherheit 0–100, Höhe in Pixeln). "
                   "Als Titel wird das größte, sicher gelesene Stück ohne Instrument genommen.")
        st.dataframe(pd.DataFrame(seiten[idx]["segmente"]), hide_index=True)
        st.text("Roher Kopfzeilen-Text:\n" + (seiten[idx]["ocr"] or "(kein Text erkannt)"))
        st.text("Roher Fußzeilen-Text:\n" + (seiten[idx]["fuss_ocr"] or "(kein Text erkannt)"))

with tab_download:
    bekannt = [e for e in eintraege if e["instrument"]]
    unbekannt = [e for e in eintraege if not e["instrument"]]

    if not bekannt:
        st.warning("Noch kein Blatt hat ein Instrument. Im Tab „Prüfen & korrigieren“ Instrument wählen.")
    else:
        st.subheader("Übersicht (Anzahl Stücke)")
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
        st.write(f"**{len(auswahl)} Stück(e)** ausgewählt")
        st.dataframe(
            pd.DataFrame([{"Nr.": e["nummer"], "Titel": e["titel"], "Stimme": e["stimme"],
                           "Tonart": e["tonart"], "Seiten": seiten_text(e),
                           "Datei": e["dateiname"]} for e in auswahl]),
            hide_index=True,
        )

        flach = set()
        zip_auswahl = erstelle_zip([(eindeutiger_pfad(e["dateiname"], flach), e["bytes"]) for e in auswahl])
        label = "alle Stimmen" if stimme_wahl == "alle" else f"Stimme {stimme_wahl}"
        st.download_button(f"📦 {instr_wahl} – {label} als ZIP herunterladen",
                           data=zip_auswahl,
                           file_name=f"{instr_wahl}_{label.replace(' ', '_')}.zip",
                           mime="application/zip")

    st.divider()
    st.subheader("Komplettarchiv")
    st.caption("Alle Stücke in der Ordnerstruktur Instrument / Stimme. "
               "Nicht erkannte Blätter liegen in _Unbekannt_Review.")
    st.download_button("📦 Alles als ZIP herunterladen",
                       data=erstelle_zip([(e["pfad"], e["bytes"]) for e in eintraege]),
                       file_name="Notenarchiv_Sortiert.zip", mime="application/zip")
    if unbekannt:
        st.caption(f"{len(unbekannt)} Datei(en) ohne Instrument liegen im Ordner _Unbekannt_Review.")
