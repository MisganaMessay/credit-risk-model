from pydantic import BaseModel

class CreditData(BaseModel):
    PricingStrategy: int
    ChannelId: int
    ProductCategory: int

class PredictionResponse(BaseModel):
    risk_probability: float
    prediction: int