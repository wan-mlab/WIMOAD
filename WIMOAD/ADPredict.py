import datetime
from WIMOAD.predict import predict
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def ADPredict():
    start_time = datetime.datetime.now()

    print("-----------------------------------------------------------------------")
    
    print("Processing:")
    
    results = predict()

    end_time = datetime.datetime.now()

    t = (end_time - start_time).total_seconds() / 60.0  # difference time in minutes

    print("Prediction completed!")
    print("-----------------------------------------------------------------------")
    print("Total running time:", t, "minutes")

    return results

