import streamlit as st
from fraude.pipelines import single_predict_pipeline
from pathlib import Path

# Initialization
TRANSASCTION_IN_PROGRESS = 'transaction_in_progress'
if TRANSASCTION_IN_PROGRESS not in st.session_state:
    st.session_state[TRANSASCTION_IN_PROGRESS] = False


def do_transaction(amount, type):
    # Placeholder for transaction logic
    st.write(f"Transaction of {amount} has been recorded as {type}.")
    st.balloons()
    st.success("Transaction recorded successfully!")
    st.session_state[TRANSASCTION_IN_PROGRESS] = False

st.write("# Welcome to the bank transaction app")
st.write("## Please enter your transaction details below")

amount = st.number_input("Transaction value", min_value=0, max_value=1000000, value=50)
type = st.selectbox("Transaction type", ["Cash in", "Cash out", "Payment", "Transfer", "Debit"])
submit = st.button("Submit")

if submit or st.session_state[TRANSASCTION_IN_PROGRESS]:
    st.session_state[TRANSASCTION_IN_PROGRESS] = True
    type_mapper = {
        "Cash in": "CASH_IN",
        "Cash out": "CASH_OUT",
        "Payment": "PAYMENT",
        "Transfer": "TRANSFER",
        "Debit": "DEBIT"
    }
    mapped_type = type_mapper.get(type, "UNKNOWN")
    if mapped_type == "UNKNOWN":
        st.error("Invalid transaction type selected.")
    else:
        project_path = Path('.').resolve()
        st.write(f"Processing transaction of {amount} as {type}...")
        y = single_predict_pipeline(project_path, amount, mapped_type)
        prediction = y[0] if len(y) > 0 else None
        if prediction is None:
            st.error("No prediction made.")
        else:
            is_fraud = prediction == 1
            if is_fraud:
                st.error("Possible fraud detected!")
                st.write(f"Transaction of {amount} has been flagged as {type}.")
                auth_code = st.text_input("An authentication code has been sent to your registered email. Please enter it below to confirm the transaction.")
                confirm_transaction = st.button("Confirm Transaction")
                if confirm_transaction:
                    if auth_code == 'correct':
                        st.success("Transaction confirmed!")
                        do_transaction(amount, type)
                    else:
                        st.error("Transaction not confirmed. Please check your authentication code.")
            else:
                st.success("No fraud detected.")
                do_transaction(amount, type)
            