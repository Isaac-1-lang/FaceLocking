"""Publish the identity-lock tracking signal to MQTT for the ESP32 needle display.

The ESP32 firmware in arduino/esp32_needle_tachometer subscribes to this topic
and drives a DC-motor needle through an L298N driver, using a tachometer for
closed-loop position feedback. See that folder's README.md for the payload
contract this module writes.
"""
import json

from paho.mqtt.client import Client


def angle_for(position):
    """Map horizontal tracking error to a 0-180 degree needle angle.

    error_x is in [-1, 1] (LEFT .. RIGHT); 0 (CENTER) maps to the needle's
    mechanical center at 90 degrees, matching the same left/right sense the
    tracker already reports.
    """
    if position is None:
        return 90.0
    return float(min(180.0, max(0.0, 90.0 + position.error_x * 90.0)))


class NeedlePublisher:
    """Thin paho-mqtt wrapper; one publish per tracked frame."""

    def __init__(self, host, port=1883, topic='facex/tracking', client_id='facex-tracker'):
        self.topic = topic
        self.client = Client(client_id=client_id)
        self.client.connect(host, port)
        self.client.loop_start()

    def publish(self, tracker, position):
        payload = {
            'state': tracker.state.name,
            'locked': tracker.state.name == 'LOCKED',
            'target': tracker.target_name,
            'horizontal': position.horizontal if position is not None else 'CENTER',
            'error_x': position.error_x if position is not None else 0.0,
            'angle': angle_for(position),
        }
        self.client.publish(self.topic, json.dumps(payload), qos=0, retain=False)

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()
