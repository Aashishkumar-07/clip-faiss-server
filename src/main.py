from services.supabase_client import initialize_supabase_client
from services.faiss_vectorstore import initialize_vectorstore
from services.mqtt_subscriber import MQTTSubscriber
from contextlib import asynccontextmanager
from predict import load_model
from fastapi import FastAPI
from pathlib import Path
import globals
import routes

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(globals.MODEL_PATH).expanduser()
    load_model(model_path)

    # Initialize vectorstore
    initialize_vectorstore()

    # Initialize supabase
    initialize_supabase_client()

    # Initialize MQTT subscriber
    mqtt_subscriber = MQTTSubscriber()
    mqtt_subscriber.start()

    print("Loaded model & Initialized vectorstore, supabase client and MQTT subscriber")

    yield

    mqtt_subscriber.stop()

# Initialization
app = FastAPI(lifespan=lifespan)
app.include_router(routes.router)

