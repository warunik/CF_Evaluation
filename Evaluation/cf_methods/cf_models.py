import numpy as np
import pandas as pd
import joblib
from Foil_Trees import domain_mappers, contrastive_explanation

# Load training data from CSV
X_train_df = pd.read_csv(r"Evaluation\split_data\adult_X_train.csv")
X_train = X_train_df.to_numpy()
feature_names = list(X_train_df.columns)

# Load pretrained model
model = joblib.load(r"Evaluation\saved_models\adult_Decision_Tree_tuned.pkl")  # update path if needed

# Set up domain mapper and explanation system
dm = domain_mappers.DomainMapperTabular(X_train, feature_names=feature_names)
exp = contrastive_explanation.ContrastiveExplanation(
    dm,
    regression=True,
    explanator=contrastive_explanation.TreeExplanator(),
    verbose=False
)

# Manual input and explanation
def manual_prediction():
    try:
        manual_input = []
        for feature in feature_names:
            value = float(input(f"Enter value for {feature}: "))
            manual_input.append(value)

        manual_sample = np.array(manual_input).reshape(1, -1)

        # Print explanation only
        explanation = exp.explain_instance_domain(model.predict, manual_sample)
        print(explanation)

    except ValueError:
        pass  # silently ignore invalid input

# Continuous prediction loop
while True:
    manual_prediction()
    if input().lower() != 'y':
        break
