from services.image_ingestion import create_embedding  
from pyaml_env import parse_config
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from pathlib import Path
import json

# Load environment variables from .env file
load_dotenv()  
BASE_DIR = Path(__file__).resolve().parent.parent

class MQTTSubscriber:
    def __init__(self, config_path: str = None):
        config_file = config_path or BASE_DIR.parent.joinpath("mqtt_config.yaml")
        self.config = parse_config(config_file)

        broker_config = self.config["broker"]
        subscriber_config = self.config["subscriber"]

        self.host = broker_config["host"]
        self.port = int(broker_config["port"])
        self.username = broker_config["username"]
        self.password = broker_config["password"]

        self.qos = int(subscriber_config["qos"])
        self.topic = subscriber_config["topic"]
        self.client_id = subscriber_config["client_id"]
        print(f"Config: {self.config}")
        
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            protocol=mqtt.MQTTv5,
            client_id=self.client_id,
        )

        self.client.username_pw_set(username=self.username, password=self.password)

        # callbacks internally called by paho-mqtt client 
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    # ===============================
    # MQTT Callbacks
    # ===============================
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            print(f"[MQTT] Connected to " f"{self.host}:{self.port}")
            client.subscribe(topic=self.topic, qos=self.qos)
            print(f"[MQTT] Subscribed to '{self.topic}'")
        else:
            print(f"[MQTT] Connection failed "f"(reason_code={reason_code})")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
            print(f"[MQTT] Unexpected disconnect " f"(reason_code={reason_code})")

    def _on_message(self, client,userdata, message):
            try:
                payload = message.payload.decode("utf-8")
                print(f"[MQTT] Message received on" f"topic='{message.topic}'")
                data = json.loads(payload)
                print(json.dumps(data, indent=2))
                self.process_message(data)

            except Exception as e:
                print(f"[MQTT] Error processing message: {e}")

    # ===============================
    # MQTT Connection Lifecycle
    # ===============================
    def start(self):
        self.client.connect(host=self.host, port=self.port, keepalive=60)
        self.client.loop_start() # Keeps main thread free by spawning a dedicated daemon thread

        # self.client.loop_forever()
    
    def stop(self):
        print("[MQTT] Stopping MQTT subscriber...")
        self.client.loop_stop()
        self.client.disconnect()

    # ===============================
    # Process message from broker
    # ===============================
    def process_message(self, payload: dict):
        print(f"[PROCESS] Received payload: {payload}")
        create_embedding(image_path=payload["image_path"],coordinate=payload["coordinate"])

# ===============================
# Quick smoke-test  
# ===============================
# if __name__ == "__main__":
#     subscriber = MQTTSubscriber()
#     subscriber.start()
