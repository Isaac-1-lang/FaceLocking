
import json
from paho.mqtt.client import Client


def angle_for(position):
    """Map horizontal face position [-1, 1] to needle angle [0, 180]."""
    if position is None:
        return 90.0

    error_x = max(-1.0, min(1.0, float(position.error_x)))
    return 90.0 + (error_x * 90.0)


class NeedlePublisher:
    def __init__(
        self,
        host,
        port=1883,
        topic="facex/tracking",
        client_id="facex-tracker",
    ):
        self.topic = topic
        self.client = Client(client_id=client_id)
        self.client.connect(host, port, keepalive=30)
        self.client.loop_start()

    def publish(self, tracker, position):
        locked = tracker.state.name == "LOCKED"

        payload = {
            "state": tracker.state.name,
            "locked": locked,
            "target": tracker.target_name,
            "horizontal": (
                position.horizontal if position is not None else "CENTER"
            ),
            "error_x": (
                float(position.error_x) if position is not None else 0.0
            ),
            "angle": angle_for(position),
        }

        result = self.client.publish(
            self.topic,
            json.dumps(payload),
            qos=0,
            retain=False,
        )

        if result.rc != 0:
            print(f"[MQTT] Publish failed: rc={result.rc}")

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()
