from data_model import TaxiRide, TaxiRidePrediction
from fastapi import FastAPI
from predict import predict, store_in_bq

app = FastAPI()


@app.get("/")
def index():
    return {"message": "NYC Taxi Ride Duration Prediction"}


@app.post("/predict", response_model=TaxiRidePrediction)
def predict_duration(data: TaxiRide):
    prediction = predict("random-forest-fare-model", data)
    return TaxiRidePrediction(**data.dict(), predicted_duration=prediction)


@app.post("/predict_batch", response_model=list[TaxiRidePrediction])
def predict_duration_batch(data_batch: list[TaxiRide]):
    predictions = []
    for data in data_batch:
        prediction = predict("random-forest-fare-model", data)
        predictions.append(
            TaxiRidePrediction(**data.dict(), predicted_duration=prediction)
        )

    return predictions


@app.post("/predict_bq", response_model=list[TaxiRidePrediction])
def predict_duration_bq(data_batch: list[TaxiRide]):
    predictions = []
    for data in data_batch:
        prediction = predict("random-forest-fare-model", data)
        prediction_full = TaxiRidePrediction(
            **data.dict(), predicted_duration=prediction
        )
        predictions.append(prediction_full)
    store_in_bq(predictions)
    return predictions
