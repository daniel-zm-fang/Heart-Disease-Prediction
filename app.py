import streamlit as st
from prediction_page import show_prediction_page
from exploration_page import show_exploration_page
from about import show_about_page


def main():
    st.set_page_config(
        page_title='Heart Diease Prediction App',
        page_icon=':heart:',
        layout='wide',
    )
    col1, col2, col3 = st.columns(spec=[1, 10, 1.4])
    with col1:
        st.image('images/logo.png', width=80)
    with col2:
        st.title('Heart Disease Prediction App')
    with col3:
        st.markdown(
            '''
            <a href='https://github.com/daniel-zm-fang/Heart-Disease-Prediction' target='_blank'>
                <img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' width='80'>
                <b>Open Source</b>
            </a>
            ''',
            unsafe_allow_html=True
        )

    tab1, tab2, tab3 = st.tabs(['Model Prediction', 'Data Exploration', 'About'])
    with tab1:
        show_prediction_page()
    with tab2:
        show_exploration_page()
    with tab3:
        show_about_page()


if __name__ == '__main__':
    main()
