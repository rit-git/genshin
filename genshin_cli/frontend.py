import hydra
from omegaconf import DictConfig

import streamlit as st
import requests

CLAIM = 'I can eat melons here.'
IMAGE = 'gs://cet-prd-dataset-images/jln/Y316832AU1/2023/05/10/Y316832AU1.jpg'

def fact_check(port):
    response = requests.post(
        f'http://localhost:{port}/api/predict',
        json={
            'claim': st.session_state['claim'],
            'image': st.session_state['image'],
        }
    )
    if response.status_code == 200:
        response = response.json()
        result = []
        for item in response:
            result.append(
                '【{}。事実確率：{:.4f}】{} '.format(
                    '事実' if item['label'] else 'ウソ', 
                    item['score'] if item['label'] else 1.0 - item['score'], 
                    item['sent'].strip()
                )
            )
        st.session_state['result'] = '\n'.join(result)

def update_textarea(key):
    st.session_state[key] = st.session_state[key]

@hydra.main(config_path="../conf", config_name="inference_config", version_base=None)
def main(cfg: DictConfig):
    port=cfg.server.port if 'server' in cfg else 13245

    st.set_page_config(
        page_title="Genshin",
        layout="wide",
    )
    st.title('幻真：幻覚を真実に')

    if 'evidence' not in st.session_state:
        st.session_state['evidence'] = IMAGE
    if 'claim' not in st.session_state:
        st.session_state['claim'] = CLAIM
    if 'result' not in st.session_state:
        st.session_state['result'] = ''

    st.text_area(
        '証拠となる画像のURLをこちらに入力してください。',
        key='evidence',
        height=200,
        on_change=update_textarea,
        kwargs={'key': 'evidence'}
    )
    st.text_area(
        '判定対象とする文章をこちらに入力してください。',
        key='claim',
        height=200,
        on_change=update_textarea,
        kwargs={'key': 'claim'}
    )
    st.text_area(
        '判定結果',
        key='result',
        height=200
    )
    st.button(
        '判定！',
        on_click=fact_check,
        kwargs={
            'port': port
        }
    )


if __name__ == '__main__':
    main()