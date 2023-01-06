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
        st.markdown(
            '''
            <a href='https://github.com/daniel-zm-fang/Heart-Disease-Prediction' target='_blank'>
                <img src='https://visualpharm.com/assets/856/Heart%20Health-595b40b65ba036ed117d4212.svg' width='80'>
            </a>
            ''',
            unsafe_allow_html=True
        )
    with col2:
        st.title('Heart Disease Prediction App')
    with col3:
        st.markdown(
            '''
            <a href='https://github.com/daniel-zm-fang/Heart-Disease-Prediction' target='_blank'>
                <img src='https://cdn-icons-png.flaticon.com/512/25/25231.png' width='80'>
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
