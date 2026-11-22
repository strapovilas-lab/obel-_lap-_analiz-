
from pathlib import Path
from datetime import datetime

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
# Lietuviškų raidžių palaikymas PDF dokumente
FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/arialbd.ttf")

pdfmetrics.registerFont(
    TTFont("Arial", str(FONT_PATH))
)

pdfmetrics.registerFont(
    TTFont("Arial-Bold", str(FONT_BOLD_PATH))
)

from PIL import Image, ImageOps


# =========================================================
# 1. PAGRINDINIAI NUSTATYMAI
# =========================================================

st.set_page_config(
    page_title="Obelų lapų DI analizė",
    page_icon="🍃",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "geriausias_modelis.keras"
SAMPLE_DIR = BASE_DIR / "pavyzdines_nuotraukos"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
]

SAMPLE_CATEGORIES = {
    "Apple_Scab": "🍂 Obelų rauplės",
    "Black_Rot": "🟤 Juodasis puvinys",
    "Apple_Rust": "🟠 Kedrinių obelų rūdys",
    "Healthy": "🌿 Sveikas lapas"
}

CLASS_INFO = {
    "Apple___Apple_scab": {
        "name": "Obelų rauplės",
        "emoji": "🍂",
        "description": (
            "Grybinė liga, dėl kurios ant lapų atsiranda "
            "tamsių alyvuogių spalvos dėmių. Stipresnio "
            "pažeidimo atveju lapai gali deformuotis ir kristi."
        ),
        "advice": (
            "Stebėkite pažeistus lapus, pašalinkite nukritusius "
            "augalų likučius ir užtikrinkite gerą vaismedžio priežiūrą."
        )
    },

    "Apple___Black_rot": {
        "name": "Juodasis puvinys",
        "emoji": "🟤",
        "description": (
            "Grybinė liga, galinti pažeisti obelų lapus, "
            "vaisius ir šakas. Ant lapų gali atsirasti "
            "rudų arba tamsių dėmių."
        ),
        "advice": (
            "Pašalinkite stipriai pažeistas augalo dalis "
            "ir stebėkite, ar liga neplinta."
        )
    },

    "Apple___Cedar_apple_rust": {
        "name": "Kedrinių obelų rūdys",
        "emoji": "🟠",
        "description": (
            "Grybinė liga, kuriai būdingos oranžinės arba "
            "geltonai oranžinės dėmės ant lapų."
        ),
        "advice": (
            "Stebėkite lapų pažeidimus ir pasirūpinkite "
            "tinkama obels priežiūra."
        )
    },

    "Apple___healthy": {
        "name": "Sveikas lapas",
        "emoji": "🌿",
        "description": (
            "Nuotraukoje modelis atpažino sveiką obels lapą. "
            "Aiškių požymių, būdingų trims mokytoms ligoms, "
            "modelis nenustatė."
        ),
        "advice": (
            "Toliau reguliariai stebėkite obels lapus ir "
            "palaikykite tinkamas augimo sąlygas."
        )
    }
}


# =========================================================
# 2. SESIJOS DUOMENYS
# =========================================================

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

if "sample_image_path" not in st.session_state:
    st.session_state.sample_image_path = None


# =========================================================
# 3. DIZAINAS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #f5f8f5;
    color: #26382d;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #204735 !important;
}

p, li, label {
    color: #344b3c;
}

.hero {
    background: linear-gradient(120deg, #e5f2e8, #f9fcf9);
    border: 1px solid #d4e5d8;
    border-radius: 20px;
    padding: 35px 30px;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 37px;
    font-weight: 800;
    margin-bottom: 10px;
    color: #204735 !important;
}

.hero p {
    color: #53675d !important;
    font-size: 17px;
    margin-bottom: 0;
}

.hero-tag {
    display: inline-block;
    background: #d6eadb;
    color: #245638;
    border-radius: 20px;
    padding: 6px 13px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 15px;
}

.section-box {
    background: #ffffff;
    border: 1px solid #dce8df;
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 15px;
}

.section-box p,
.section-box li {
    color: #344b3c !important;
}

.result-box {
    background: #ffffff;
    border: 1px solid #cbded0;
    border-radius: 16px;
    padding: 24px;
    margin-top: 10px;
}

.result-box h2 {
    color: #204735 !important;
    margin-top: 0;
}

.result-score {
    font-size: 29px;
    font-weight: 800;
    color: #286343;
    margin-top: 12px;
}

.healthy-card {
    background: #e8f5e9;
    border: 1px solid #b9d8bd;
    border-radius: 13px;
    padding: 18px;
}

.healthy-card h3,
.healthy-card p {
    color: #204735 !important;
}

.disease-card {
    background: #ffffff;
    border: 1px solid #dce8df;
    border-radius: 13px;
    padding: 18px;
    min-height: 190px;
}

.disease-card h3 {
    color: #204735 !important;
    margin-top: 0;
}

.disease-card p {
    color: #344b3c !important;
}

.recommendation-box {
    background: #e8f5e9;
    color: #204735;
    border: 1px solid #b9d8bd;
    border-radius: 11px;
    padding: 17px;
    margin-top: 10px;
}

.recommendation-box p {
    color: #204735 !important;
}

.warning-box {
    background: #fff4df;
    color: #684817;
    border: 1px solid #e8cf9d;
    border-radius: 11px;
    padding: 17px;
    margin-top: 12px;
}

.warning-box p,
.warning-box strong {
    color: #684817 !important;
}

.stButton > button {
    background: #286343;
    color: #ffffff !important;
    border: none;
    border-radius: 10px;
    min-height: 43px;
    font-weight: 650;
}

.stButton > button:hover {
    background: #1d4d33;
    color: #ffffff !important;
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 2px dashed #8ab69a;
    border-radius: 14px;
    padding: 16px;
}

[data-testid="stFileUploader"] section {
    background: #ffffff;
}

[data-testid="stFileUploader"] * {
    color: #204735 !important;
}

[data-testid="stFileUploader"] button {
    background: #286343 !important;
    color: #ffffff !important;
    border: none !important;
}

[data-testid="stFileUploader"] button * {
    color: #ffffff !important;
}

.footer {
    text-align: center;
    color: #718176 !important;
    font-size: 13px;
    padding-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. MODELIO ĮKĖLIMAS
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error("Nepavyko įkelti DI modelio.")
    st.exception(e)
    st.stop()

# =========================================================
# PDF ATASKAITOS GENERAVIMAS
# =========================================================

def create_pdf_report(result, image):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontName="Arial-Bold",
        fontSize=19,
        leading=25,
        textColor=colors.HexColor("#204735"),
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontName="Arial-Bold",
        textColor=colors.HexColor("#286343"),
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontName="Arial",
        fontSize=10,
        leading=15
    )

    story = []

    # Antraštė
    story.append(
        Paragraph(
            "Obelų lapų būklės vertinimo ataskaita",
            title_style
        )
    )

    story.append(Spacer(1, 10))

    # Pagrindinė informacija
    story.append(
        Paragraph("Analizės informacija", heading_style)
    )

    story.append(
        Paragraph(
            f"Nuotraukos pavadinimas: {result['filename']}",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"Analizės data ir laikas: {result['time']}",
            normal_style
        )
    )

    story.append(Spacer(1, 12))

    # Nuotrauka
    story.append(
        Paragraph("Analizuota nuotrauka", heading_style)
    )

    image_buffer = BytesIO()
    image.save(image_buffer, format="JPEG", quality=90)
    image_buffer.seek(0)

    pdf_image = PDFImage(
        image_buffer,
        width=8 * cm,
        height=8 * cm,
        kind="proportional"
    )

    story.append(pdf_image)
    story.append(Spacer(1, 12))

    # Prognozės rezultatas
    predicted_class = result["predicted_class"]
    class_info = CLASS_INFO[predicted_class]

    story.append(
        Paragraph("Atpažinimo rezultatas", heading_style)
    )

    story.append(
        Paragraph(
            f"Nustatyta būklė: <b>{class_info['name']}</b>",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"Modelio prognozės tikimybė: "
            f"<b>{result['confidence'] * 100:.2f} %</b>",
            normal_style
        )
    )

    story.append(Spacer(1, 10))

    # Visų klasių tikimybės
    story.append(
        Paragraph("Visų klasių tikimybės", heading_style)
    )

    table_data = [
        ["Obels lapo būklė", "Tikimybė"]
    ]

    for i, class_name in enumerate(CLASS_NAMES):

        probability = result["probabilities"][i] * 100

        table_data.append([
            CLASS_INFO[class_name]["name"],
            f"{probability:.2f} %"
        ])

    probability_table = Table(
        table_data,
        colWidths=[11 * cm, 5 * cm]
    )

    probability_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#286343")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Arial"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1),
             colors.HexColor("#F3F8F4")),
            ("PADDING", (0, 0), (-1, -1), 8)
        ])
    )

    story.append(probability_table)
    story.append(Spacer(1, 15))

    # Rekomendacija
    story.append(
        Paragraph("Rekomendacija", heading_style)
    )

    story.append(
        Paragraph(
            class_info["advice"],
            normal_style
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Pastaba: DI modelio pateiktas rezultatas yra "
            "pagalbinė informacija, o ne galutinė augalo ligos "
            "diagnozė. Rekomenduojama rezultatą patikrinti vizualiai.",
            normal_style
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()



# =========================================================
# 5. PAGRINDINIS EKRANAS
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">DIRBTINIO INTELEKTO PROJEKTAS</div>
    <h1>🍃 Obelų lapų būklės vertinimas</h1>
    <p>
        Išmanioji sistema, kuri analizuoja obelų lapų nuotraukas
        ir padeda atpažinti galimas ligas naudojant dirbtinį intelektą.
    </p>
</div>
""", unsafe_allow_html=True)


metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Modelio testavimo tikslumas", "89,08 %")

with metric2:
    st.metric("Atpažįstamos klasės", "4")

with metric3:
    st.metric("Nuotraukos dydis", "224 × 224 px")


st.divider()


# =========================================================
# 6. NUOTRAUKOS ĮKĖLIMAS
# =========================================================

left_column, right_column = st.columns([1.1, 0.9])

with left_column:

    st.subheader("📷 Įkelkite savo nuotrauką")

    uploaded_file = st.file_uploader(
        "Pasirinkite JPG, JPEG arba PNG formato nuotrauką",
        type=["jpg", "jpeg", "png"],
        key=f"leaf_uploader_{st.session_state.upload_key}"
    )

    st.caption(
        "Geriausiam rezultatui rekomenduojama aiški, "
        "gerai apšviesta vieno obels lapo nuotrauka."
    )

    with st.expander("💡 Kaip padaryti tinkamą nuotrauką?"):
        st.markdown("""
        - Fotografuokite lapą iš arti.
        - Pasirūpinkite geru apšvietimu.
        - Venkite stiprių šešėlių.
        - Stenkitės, kad lapas būtų ryškus.
        - Geriausia, kai nuotraukoje matomas visas lapas.
        """)


with right_column:

    st.subheader("🔍 Sistemos galimybės")

    st.write(
        "Dirbtinio intelekto sistema analizuoja įkeltą obels lapo "
        "nuotrauką ir priskiria ją vienai iš keturių kategorijų:"
    )

    st.markdown("""
    - 🍂 **Obelų rauplės**
    - 🟤 **Juodasis puvinys**
    - 🟠 **Kedrinių obelų rūdys**
    - 🌿 **Sveikas obels lapas**
    """)

    st.write(
        "Atlikus analizę, pateikiama labiausiai tikėtina klasė, "
        "kiekvienos klasės tikimybės ir rekomendacijos."
    )

    st.write(
        "Sistema taip pat leidžia peržiūrėti pavyzdines nuotraukas "
        "ir atsisiųsti analizės ataskaitą PDF formatu."
    )




# =========================================================
# 7. PAVYZDINIŲ NUOTRAUKŲ GALERIJA
# =========================================================

st.divider()

st.header("🖼️ Pavyzdinių nuotraukų galerija")

st.write(
    "Pasirinkite vieną iš paruoštų obelų lapų nuotraukų. "
    "Sistema ją išanalizuos taip pat, kaip jūsų įkeltą nuotrauką."
)

if SAMPLE_DIR.exists():

    for folder_name, category_title in SAMPLE_CATEGORIES.items():

        category_folder = SAMPLE_DIR / folder_name

        with st.expander(category_title, expanded=False):

            if category_folder.exists():

                sample_files = sorted([
                    file for file in category_folder.iterdir()
                    if file.is_file()
                    and file.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ])

                if not sample_files:

                    st.info("Šiame aplanke nuotraukų nerasta.")

                else:

                    sample_columns = st.columns(4)

                    for index, sample_path in enumerate(sample_files):

                        with sample_columns[index % 4]:

                            try:

                                sample_image = Image.open(sample_path)
                                sample_image = ImageOps.exif_transpose(
                                    sample_image
                                ).convert("RGB")

                                st.image(
                                    sample_image,
                                    width="stretch"
                                )

                                st.caption(sample_path.name)

                                if st.button(
                                    "Pasirinkti nuotrauką",
                                    key=f"sample_{folder_name}_{index}",
                                    width="stretch"
                                ):

                                    st.session_state.sample_image_path = str(
                                        sample_path
                                    )

                                    st.session_state.last_result = None

                                    # Išvalomas ankstesnis įkeltas failas.
                                    st.session_state.upload_key += 1

                                    st.rerun()

                            except Exception:

                                st.warning(
                                    f"Nepavyko atidaryti {sample_path.name}"
                                )

            else:

                st.warning(
                    f"Nerastas aplankas: {category_folder}"
                )

else:

    st.warning(
        f"Nerastas pavyzdinių nuotraukų aplankas: {SAMPLE_DIR}"
    )


# =========================================================
# 8. PASIRINKTOS NUOTRAUKOS PARUOŠIMAS
# =========================================================

active_image = None
active_name = None
active_source = None

# Jei vartotojas įkėlė savo nuotrauką,
# ji turi pirmenybę prieš galerijos nuotrauką.

if uploaded_file is not None:

    st.session_state.sample_image_path = None

    try:

        active_image = Image.open(uploaded_file)
        active_image = ImageOps.exif_transpose(
            active_image
        ).convert("RGB")

        active_name = uploaded_file.name

        active_source = (
            f"upload_{st.session_state.upload_key}_{uploaded_file.name}"
        )

    except Exception as e:

        st.error("Nepavyko atidaryti įkeltos nuotraukos.")
        st.exception(e)

elif st.session_state.sample_image_path is not None:

    sample_path = Path(st.session_state.sample_image_path)

    if sample_path.exists():

        try:

            active_image = Image.open(sample_path)
            active_image = ImageOps.exif_transpose(
                active_image
            ).convert("RGB")

            active_name = sample_path.name
            active_source = f"sample_{sample_path}"

        except Exception as e:

            st.error("Nepavyko atidaryti pavyzdinės nuotraukos.")
            st.exception(e)

    else:

        st.warning("Pasirinkta pavyzdinė nuotrauka neberasta.")


# =========================================================
# 9. NUOTRAUKOS ANALIZĖ
# =========================================================

if active_image is not None:

    st.divider()

    st.subheader("🔬 Pasirinktos nuotraukos analizė")

    image_column, analysis_column = st.columns([1, 1])

    with image_column:

        st.write(f"**Nuotrauka:** {active_name}")

        st.image(
            active_image,
            width="stretch"
        )

    with analysis_column:

        st.write("**Dirbtinio intelekto analizė**")

        if st.button(
            "🔬 Analizuoti nuotrauką",
            width="stretch",
            key="analyze_button"
        ):

            with st.spinner(
                "Dirbtinis intelektas analizuoja nuotrauką..."
            ):

                resized_image = active_image.resize(IMAGE_SIZE)

                img_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # Modelis pats atlieka normalizavimą.
                # Papildomai dalinti iš 255 nereikia.

                predictions = model.predict(
                    img_array,
                    verbose=0
                )

                predicted_index = int(
                    np.argmax(predictions[0])
                )

                predicted_class = CLASS_NAMES[predicted_index]

                confidence = float(
                    predictions[0][predicted_index]
                )

                analysis_time = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                result_data = {
                    "filename": active_name,
                    "source": active_source,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": [
                        float(value) for value in predictions[0]
                    ],
                    "time": analysis_time
                }

                st.session_state.last_result = result_data

                class_data = CLASS_INFO[predicted_class]

                st.session_state.analysis_history.append({
                    "Nuotrauka": active_name,
                    "Nustatyta būklė": class_data["name"],
                    "Tikimybė": f"{confidence * 100:.2f} %",
                    "Laikas": analysis_time
                })

    # Naujos analizės mygtukas

    if st.button(
        "➕ Pradėti naują analizę",
        width="stretch",
        key="new_analysis_button"
    ):

        st.session_state.last_result = None
        st.session_state.sample_image_path = None
        st.session_state.upload_key += 1

        st.rerun()

    # Rodome rezultatą tik tos pačios nuotraukos analizei.

    result = st.session_state.last_result

    if result is not None and result["source"] == active_source:

        st.divider()

        class_data = CLASS_INFO[result["predicted_class"]]

        st.subheader("📊 Analizės rezultatas")

        st.markdown(
            f"""
            <div class="result-box">
                <h2>{class_data["emoji"]} {class_data["name"]}</h2>
                <p>{class_data["description"]}</p>
                <div class="result-score">
                    {result["confidence"] * 100:.2f} %
                </div>
                <p>Modelio prognozės tikimybė</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(result["confidence"])

        st.markdown("**Rekomendacija:**")

        st.markdown(
            f"""
            <div class="recommendation-box">
                <p>{class_data["advice"]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if result["predicted_class"] == "Apple___healthy":

            st.markdown("""
            <div class="healthy-card">
                <h3>🌿 Sveikas lapas</h3>
                <p>
                    Modelis nuotraukoje atpažino sveiką obels lapą.
                    Toliau stebėkite augalo būklę ir reguliariai
                    apžiūrėkite lapus.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Galimos ligos požymiai</strong>
                <p>
                    Modelis aptiko požymių, būdingų galimai ligai.
                    Rezultatą rekomenduojama patikrinti vizualiai.
                </p>
            </div>
            """, unsafe_allow_html=True)

        
        st.markdown("#### Visų klasių tikimybės")

        # Paruošiame duomenis diagramai
        chart_data = pd.DataFrame({
            "Klasė": [
                CLASS_INFO[class_name]["name"]
                for class_name in CLASS_NAMES
            ],
            "Tikimybė (%)": [
                probability * 100
                for probability in result["probabilities"]
            ]
        })

        # Atvaizduojame horizontalią diagramą
        st.bar_chart(
            chart_data,
            x="Klasė",
            y="Tikimybė (%)",
            horizontal=True,
            height=250
        )

        # Tikslūs procentai po diagrama
        for _, row in chart_data.iterrows():

            st.write(
                f"**{row['Klasė']}:** "
                f"{row['Tikimybė (%)']:.2f} %"
            )

        st.caption(
            f"Nuotrauka: {result['filename']} | "
            f"Analizės laikas: {result['time']}"
        )

        # PDF ataskaitos generavimas
        pdf_report = create_pdf_report(
            result,
            active_image
        )

        st.download_button(
            label="📄 Atsisiųsti analizės ataskaitą (PDF)",
            data=pdf_report,
            file_name="obelu_lapu_analizes_ataskaita.pdf",
            mime="application/pdf",
            width="stretch"
        )


else:

    st.markdown("""
    <div class="section-box">
        <h3>👋 Pradėkite nuo nuotraukos pasirinkimo</h3>
        <p>
            Galite įkelti savo obels lapo nuotrauką iš kompiuterio
            arba pasirinkti vieną iš galerijoje esančių pavyzdžių.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 10. ANALIZĖS ISTORIJA
# =========================================================

st.divider()

st.header("📋 Analizės istorija")

st.write(
    "Čia pateikiami šios programos sesijos metu "
    "atliktų analizių rezultatai."
)

if st.session_state.analysis_history:

    history_column, button_column = st.columns([3, 1])

    with history_column:

        st.metric(
            "Atliktų analizių skaičius",
            len(st.session_state.analysis_history)
        )

    with button_column:

        if st.button("🗑️ Išvalyti istoriją"):

            st.session_state.analysis_history = []
            st.session_state.last_result = None

            st.rerun()

    st.dataframe(
        st.session_state.analysis_history,
        width="stretch",
        hide_index=True
    )

    st.caption(
        "Istorija saugoma tik dabartinės programos sesijos metu. "
        "Uždarius sesiją, ji gali būti prarasta."
    )

else:

    st.markdown("""
    <div class="recommendation-box">
        <p>
            📋 Analizės istorija kol kas tuščia.
            Atlikus pirmąją analizę, rezultatai atsiras čia.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 11. ATPAŽĮSTAMOS OBELŲ LAPŲ BŪKLĖS
# =========================================================

st.divider()

st.header("🌱 Atpažįstamos obelų lapų būklės")

disease_columns = st.columns(4)

for column, class_name in zip(disease_columns, CLASS_NAMES):

    info = CLASS_INFO[class_name]

    with column:

        if class_name == "Apple___healthy":

            st.markdown(
                f"""
                <div class="healthy-card">
                    <h3>{info["emoji"]} {info["name"]}</h3>
                    <p>{info["description"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="disease-card">
                    <h3>{info["emoji"]} {info["name"]}</h3>
                    <p>{info["description"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# 12. INFORMACIJA APIE MODELĮ
# =========================================================

st.divider()

st.header("🤖 Informacija apie DI modelį")

info1, info2 = st.columns([1, 1])

with info1:

    st.markdown("""
    <div class="section-box">
        <h3>Modelio architektūra</h3>
        <p>
            Naudojamas konvoliucinis neuroninis tinklas,
            paremtas <b>MobileNetV2</b> architektūra.
        </p>
        <p>
            Nuotraukos prieš analizę pakeičiamos į
            224 × 224 pikselių dydį.
        </p>
    </div>
    """, unsafe_allow_html=True)

with info2:

    st.markdown("""
    <div class="section-box">
        <h3>Modelio rezultatai</h3>
        <p>
            Testavimo tikslumas – <b>89,08 %</b>.
        </p>
        <p>
            Modelis apmokytas atpažinti keturias obelų lapų klases.
        </p>
    </div>
    """, unsafe_allow_html=True)



# =========================================================
# 13. DUOMENŲ ŠALTINIAI IR PRIVATUMAS
# =========================================================

with st.expander("📚 Duomenų šaltiniai ir privatumas"):

    st.markdown("""
    Modelio mokymui naudotos obelų lapų nuotraukos,
    suskirstytos į keturias klases.

    Pagrindinis duomenų šaltinis:
    `timm/plant-pathology-2021` (Hugging Face).

    Papildomai naudotos juodojo puvinio nuotraukos.

    Įkeltos nuotraukos naudojamos analizei.
    Ši programa nenumato jų saugojimo atskiroje duomenų bazėje.

    Analizės istorijoje saugomi failo pavadinimas,
    nustatyta būklė, tikimybė ir analizės laikas.
    """)


# =========================================================
# 14. PUSLAPIO APAČIA
# =========================================================

st.markdown("""
<div class="footer">
    Obelų lapų būklės vertinimo sistema |
    Dirbtinio intelekto projektas
</div>
""", unsafe_allow_html=True)
