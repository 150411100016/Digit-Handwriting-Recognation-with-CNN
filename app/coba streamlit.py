import streamlit as st

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

# SIDEBAR
st.sidebar.image("assets/img/digitalent.png", width=100)
st.sidebar.title ("Digit Handwriting Recognation with CNN")

# # CONTENT
# st.write("Hello ,let's learn how to build a streamlit app together")
# st.title ("this is the app title")
# st.header("this is the markdown")
# st.markdown("this is the header")
# st.subheader("this is the subheader")
# st.caption("this is the caption")
# st.code("x=2021")
# st.latex(r''' a+a r^1+a r^2+a r^3 ''')
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

app_mode = st.sidebar.selectbox('Select Menu',['Home','Requirement','Dataset','Prepare Model','Predict Result','Improvement'])
if app_mode=='Home':    
    st.title('LOAN PREDICTION :')      
elif app_mode == 'Requirement':      
    st.title('SECOND PREDICTION :')   

# if st.sidebar.button('Click Me', key='my_button', help='Click this button to perform an action'):
#    st.write('You clicked the button!')  

# if st.sidebar.button('Another Me', key='my_buttons', help='Click this button to perform an action'):
#    st.write('You clicked the button!') 

# local_css("assets/css/style.css")

# with st.sidebar:
#     st.button("button sidebar 1")
#     st.button("button sidebar longer text")
#     st.button("button sidebar 2")
#     st.button("button sidebar 3")
    