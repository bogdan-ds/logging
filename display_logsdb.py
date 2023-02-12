import base64
import os
import sqlite3

import streamlit as st
import pandas as pd

from typing import Optional
from PIL import Image
from io import BytesIO


@st.cache
def load_data() -> str:
    conn = create_connection("logtrucks.db")
    query = "SELECT * FROM detections"
    query = conn.execute(query)
    cols = [column[0] for column in query.description]
    results_df = pd.DataFrame.from_records(
        data=query.fetchall(),
        columns=cols
    )
    results_df = add_images_column_to_df(results_df)
    return results_df.to_html(escape=False,
                              formatters=dict(thumbnail=image_formatter))


def create_connection(db_file: str) -> sqlite3.Connection:
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        st.write(e)

    return conn


def add_images_column_to_df(df: pd.DataFrame) -> pd.DataFrame:
    df["images"] = df["id"].map(image_column)
    return df


def image_column(uuid: str) -> Optional[str]:
    images = find_images_from_uuid(uuid, "results")
    if not images:
        return None
    image = image_formatter(f"results/{images[0]}")
    return image


def find_images_from_uuid(uuid: str, results_dir: str) -> list:
    files = [file for file in os.listdir(results_dir)
             if not file.endswith(".lpr.jpg")]
    images = [filename for filename in files
              if filename.split("_")[0] == uuid]
    return images


def image_formatter(img_path: str) -> str:
    return f'<img src="data:image/png;base64,{image_to_base64(img_path)}">'


def image_to_base64(img_path: str) -> str:
    img = get_thumbnail(img_path)
    with BytesIO() as buffer:
        img.save(buffer, 'png')
        return base64.b64encode(buffer.getvalue()).decode()


def get_thumbnail(path: str) -> Image:
    img = Image.open(path)
    img.thumbnail((100, 100))
    return img


st.set_page_config(layout="wide")
html = load_data()
st.markdown(
    html,
    unsafe_allow_html=True
)
