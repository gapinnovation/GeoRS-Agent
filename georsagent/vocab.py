"""Closed vocabularies: WHU-RS19 scene classes and WHU-RS19 ABZSL attributes.

The Italian names are the column/class names used in the annotation workbook
(sheet "40 classi REF"); the English ids are the canonical identifiers used by
every tool, prompt and log. Nothing outside these lists is admissible.
"""

# Italian label in workbook -> canonical WHU-RS19 label (folder name in Filepath)
CLASS_MAP = {
    "Aeroporto": "Airport",
    "Spiaggia": "Beach",
    "Ponte": "Bridge",
    "Zona commerciale": "Commercial",
    "Deserto": "Desert",
    "Campo": "Farmland",
    "Campo da calcio": "footballField",
    "Foresta": "Forest",
    "Zona industriale": "Industrial",
    "Prato": "Meadow",
    "Montagna": "Mountain",
    "Parco": "Park",
    "Parcheggio": "Parking",
    "Stagno": "Pond",
    "Porto": "Port",
    "Stazione ferroviaria": "railwayStation",
    "Zona residenziale": "Residential",
    "Fiume": "River",
    "Viadotto": "Viaduct",
}
CLASSES = sorted(CLASS_MAP.values(), key=str.lower)

# Italian column -> (english id, group, short visual definition shown to the VLM)
ATTRIBUTES = {
    "Aereo": ("airplane", "object", "aircraft visible on the ground or in the air"),
    "Strada": ("road", "object", "paved or unpaved road"),
    "Ponte": ("bridge", "object", "bridge structure crossing water, road or rail"),
    "Mezzi di trasporto su strada": ("road_vehicle", "object", "cars, trucks or buses"),
    "Case": ("house", "object", "small detached residential houses"),
    "Prato": ("grass", "object", "grass or lawn surface"),
    "Alberi": ("tree", "object", "individual trees or tree canopy"),
    "Monti": ("mountain", "object", "mountain or steep relief"),
    "Terreno": ("bare_soil", "object", "bare soil or unvegetated ground"),
    "Acqua": ("water", "object", "water body: river, lake, sea, pond"),
    "Sabbia": ("sand", "object", "sand surface"),
    "Mezzi di trasporto via acqua": ("watercraft", "object", "boats or ships"),
    "Banchina": ("dock", "object", "quay, dock or pier"),
    "Stadio": ("stadium", "object", "stadium or stands around a sports field"),
    "Campo da calcio": ("football_pitch", "object", "football pitch with markings"),
    "Edifici": ("building", "object", "large non-residential or multi-storey buildings"),
    "Posti per auto": ("parking_space", "object", "marked parking spaces or parking lots"),
    "Binari": ("railway_track", "object", "railway tracks"),
    "Treni": ("train", "object", "trains or rail cars"),
    "Rosso": ("red", "color", "red"),
    "Arancione": ("orange", "color", "orange"),
    "Ocra": ("ochre", "color", "ochre"),
    "Beige": ("beige", "color", "beige"),
    "Verde": ("green", "color", "green"),
    "Azzurro": ("light_blue", "color", "light blue"),
    "Blu": ("blue", "color", "blue"),
    "Marrone": ("brown", "color", "brown"),
    "Grigio": ("grey", "color", "grey"),
    "Bianco": ("white", "color", "white"),
    "Nero": ("black", "color", "black"),
    "Rettangolo": ("rectangle", "shape", "rectangular shapes"),
    "Triangolo": ("triangle", "shape", "triangular shapes"),
    "Quadrato": ("square", "shape", "square shapes"),
    "Sinusoide": ("sinusoid", "shape", "sinuous / meandering lines"),
    "Linea": ("line", "shape", "straight lines"),
    "Linea tratteggiata": ("dashed_line", "shape", "dashed lines (e.g. road markings)"),
    "Curva": ("curve", "shape", "open curves"),
    "Curva chiusa": ("closed_curve", "shape", "closed curves (circles, ovals, loops)"),
}
ATTR_IDS = [v[0] for v in ATTRIBUTES.values()]
ATTR_GROUP = {v[0]: v[1] for v in ATTRIBUTES.values()}
ATTR_DEF = {v[0]: v[2] for v in ATTRIBUTES.values()}
IT2ATTR = {k: v[0] for k, v in ATTRIBUTES.items()}

# Three-valued evidence state. "uncertain" is treated exactly like "not provided":
# it never counts as evidence of absence (reviewer C1.8: omitted != negated).
PRESENT, ABSENT, UNCERTAIN = "present", "absent", "uncertain"
