{\rtf1\ansi\ansicpg1252\cocoartf2707
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 INSTR_KEYS = sorted(INSTRUMENTE_MAP, key=len, reverse=True)   # l\'e4ngste zuerst\
INSTR_ALT = "|".join(re.escape(k) for k in INSTR_KEYS)\
\
def parse_header_text(ocr_text):\
    lines = [l.strip() for l in ocr_text.split("\\n") if l.strip()]\
    full_text = " ".join(lines)\
    data = \{"nummer": None, "instrument": None, "stimme": None, "tonart": None, "titel": None\}\
\
    # 1. Nummer\
    m = re.search(r'\\b(?:Nr|No)\\.?\\s*(\\d\{1,3\})\\b', full_text, re.I)\
    if m:\
        data["nummer"] = m.group(1).zfill(3)\
\
    # 2. Instrument (Treffer merken, um die Zeile sp\'e4ter auszuschlie\'dfen)\
    m = re.search(rf'\\b(\{INSTR_ALT\})\\b', full_text, re.I)\
    if m:\
        data["instrument"] = INSTRUMENTE_MAP[m.group(1).lower()]\
\
        # 3. Stimme nur direkt vor/nach dem Instrument\
        v = (re.search(rf'\\b([1-4])\\s*\\.?\\s*\{re.escape(m.group(1))\}\\b', full_text, re.I)\
             or re.search(rf'\\b\{re.escape(m.group(1))\}\\s*([1-4])\\b', full_text, re.I))\
        if v:\
            data["stimme"] = v.group(1)\
\
    # 4. Tonart\
    t = re.search(r'\\bin\\s+(Es|As|B|C|D|F|G)\\b', full_text)\
    if t:\
        data["tonart"] = t.group(1)\
\
    # 5. Titel: Zeilen mit Instrument/Stimme/Tonart \'fcberspringen\
    candidates = []\
    for line in lines:\
        if re.search(rf'\\b(\{INSTR_ALT\})\\b', line, re.I):\
            continue\
        c = re.sub(r'\\b(?:Nr|No)\\.?\\s*\\d\{1,3\}\\b', '', line, flags=re.I)\
        c = re.sub(r'[^\\w\\s-]', '', c).strip()\
        if len(c) > 2:\
            candidates.append(c)\
    if candidates:\
        data["titel"] = max(candidates, key=len).replace(" ", "_")\
    return data}