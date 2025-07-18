import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from pipelines.lead_conversion_rate.steps.model_fit import fit, predict
from pipelines.lead_conversion_rate.model.utls.evaluation import Evaluation



if __name__ == '__main__':
    fit()

    X_test = pd.read_pickle("pipelines/lead_conversion_rate/model/pickles/X_test.pkl")
    y_test = pd.read_pickle("pipelines/lead_conversion_rate/model/pickles/y_test.pkl")
    #
    total = 10
    X_test = X_test[:total].to_dict(orient='records')
    y_test = y_test[:total]
    y_pred_init = predict('init_stage', data=X_test)

    #
    y_pred_mid = predict('mid_stage')

    y_pred_final = predict('final_stage', data=X_test)

    print("y_pred_init", y_pred_init, y_test.values)
    print("y_pred_mid", y_pred_mid, y_test.values)
    print("y_pred_final", y_pred_final, y_test.values)

    #Evaluation().create_classification_report('init_stage', y_test_slice, y_pred_init.probabilities, show_save=True)
    #Evaluation().create_classification_report('mid_stage', y_test_slice, y_pred_mid.probabilities, show_save=True)
    #Evaluation().create_classification_report('final_stage', y_test_slice, y_pred_final.probabilities, show_save=True)