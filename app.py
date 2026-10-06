"""
Gradio Web Application for XGBoost Model Inference
Predicts target classification from Feature 1 and Feature 2 inputs.
Designed for both local execution and Hugging Face Spaces deployment.
"""

import os
import joblib
import pandas as pd
import gradio as gr

# Resolve model path relative to app.py
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file 'model.pkl' not found at '{MODEL_PATH}'. "
        "Please run 'python mlflow_demo.py' or 'python export_model.py' to generate model.pkl."
    )

model = joblib.load(MODEL_PATH)


def predict(feature1, feature2):
    """
    Takes Feature 1 and Feature 2 inputs, formats them as a DataFrame,
    and returns the predicted class and confidence probability.
    """
    if feature1 is None or feature2 is None:
        return "Please provide valid numbers for both Feature 1 and Feature 2."

    try:
        # Create input DataFrame with exact training feature names
        input_df = pd.DataFrame(
            [[float(feature1), float(feature2)]],
            columns=["feature1", "feature2"]
        )

        # Generate prediction
        prediction = model.predict(input_df)
        pred_class = int(prediction[0])

        # Obtain probability estimates if available
        confidence_info = ""
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            confidence_info = (
                f"\n\nClass Probabilities:\n"
                f" - Class 0: {probabilities[0]:.2%}\n"
                f" - Class 1: {probabilities[1]:.2%}"
            )

        return f"Predicted Class: {pred_class}{confidence_info}"
    except Exception as e:
        return f"Prediction Error: {str(e)}"


# Build Gradio Interface
with gr.Blocks(title="XGBoost Model Prediction") as demo:
    gr.Markdown("# 🚀 XGBoost Classification Model Prediction")
    gr.Markdown(
        "This interactive application predicts the target classification (Class 0 or Class 1) "
        "using a trained **XGBoost Classifier** tracked with MLflow. "
        "Enter values for **Feature 1** and **Feature 2** below, then click **Predict**."
    )

    with gr.Row():
        with gr.Column():
            feature1_input = gr.Number(
                label="Feature 1",
                value=4.0,
                info="Numeric value for Feature 1 (e.g., 1 to 10)"
            )
            feature2_input = gr.Number(
                label="Feature 2",
                value=10.0,
                info="Numeric value for Feature 2 (e.g., 11 to 20)"
            )
            with gr.Row():
                predict_button = gr.Button("Predict", variant="primary")
                clear_button = gr.ClearButton(
                    components=[feature1_input, feature2_input],
                    value="Clear Inputs"
                )

        with gr.Column():
            result_output = gr.Textbox(
                label="Prediction Result",
                interactive=False,
                lines=4,
                placeholder="The prediction result will appear here..."
            )

    gr.Markdown("### Sample Test Cases")
    gr.Examples(
        examples=[
            [4.0, 10.0],
            [2.0, 12.0],
            [7.0, 17.0],
            [9.0, 19.0],
            [1.0, 11.0],
            [10.0, 20.0]
        ],
        inputs=[feature1_input, feature2_input]
    )

    predict_button.click(
        fn=predict,
        inputs=[feature1_input, feature2_input],
        outputs=result_output
    )


if __name__ == "__main__":
    # In Hugging Face Spaces or containers, bind to 0.0.0.0; locally bind to 127.0.0.1
    server_port = int(os.environ.get("PORT", 7860))
    server_name = "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    print(f"Launching Gradio app on http://{server_name}:{server_port}")
    demo.launch(server_name=server_name, server_port=server_port)
