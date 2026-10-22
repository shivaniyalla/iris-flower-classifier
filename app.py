
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42,
    stratify=iris.target
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.title("🌸 Iris Flower Classification")
st.write("Enter flower measurements to predict its species.")

st.metric("Model Accuracy", f"{accuracy:.0%}")

st.subheader("Flower Measurements")

sepal_length = st.number_input("Sepal Length (cm)", 0.1, 10.0, 5.1)
sepal_width = st.number_input("Sepal Width (cm)", 0.1, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.1, 10.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.1, 10.0, 0.2)

if st.button("Predict Flower 🌼"):
    values = [[sepal_length, sepal_width, petal_length, petal_width]]
    result = model.predict(values)[0]
    st.success(f"Predicted Flower: {iris.target_names[result].title()}")
