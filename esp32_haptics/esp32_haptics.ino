// PROJECT ANGELINA — Neural-Ergonomic Focus Bot
// ============================================================
// SCRIPT: ESP32 HAPTIC FEEDBACK RECEIVER
// Purpose: Listens to the Serial port at 115200 baud for "TRIGGER\n" 
//          or "WARNING\n" from the Python inference engine, and activates 
//          the vibration motor and buzzer.
// ============================================================

// Pin Assignments
const int MOTOR_PIN = 18;  // Connect Vibration Motor + to GPIO18
const int BUZZER_PIN = 19; // Connect Buzzer + to GPIO19
                           // Connect - of both to GND

void setup() {
  // Start serial communication at 115200 baud (must match Python script)
  Serial.begin(115200);
  
  // Configure output pins
  pinMode(MOTOR_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  // Ensure they start OFF
  digitalWrite(MOTOR_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  // Quick double-buzz to confirm it's alive and not stuck in reset!
  digitalWrite(BUZZER_PIN, HIGH); delay(50); digitalWrite(BUZZER_PIN, LOW); delay(100);
  digitalWrite(BUZZER_PIN, HIGH); delay(50); digitalWrite(BUZZER_PIN, LOW);

  Serial.println("ESP32 Haptic Receiver Ready.");
}

void loop() {
  // Check if data is available from Python script
  if (Serial.available() > 0) {
    // Read the incoming string until newline character
    String command = Serial.readStringUntil('\n');
    command.trim(); // Remove any carriage returns or whitespace

    if (command == "TRIGGER") {
      Serial.println("ACK: TRIGGER RECEIVED");
      playTriggerAlert();
    } 
    else if (command == "WARNING") {
      Serial.println("ACK: WARNING RECEIVED");
      playWarningAlert();
    }
  }
}

// ----------------------------------------------------
// Alert Patterns
// ----------------------------------------------------

void playTriggerAlert() {
  // A strong, urgent alert for sustained bad posture
  
  // Pulse 1
  digitalWrite(MOTOR_PIN, HIGH);
  digitalWrite(BUZZER_PIN, HIGH);
  delay(300);
  
  // Pause
  digitalWrite(MOTOR_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  delay(150);
  
  // Pulse 2
  digitalWrite(MOTOR_PIN, HIGH);
  digitalWrite(BUZZER_PIN, HIGH);
  delay(300);
  
  // Pause
  digitalWrite(MOTOR_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  delay(150);
  
  // Pulse 3
  digitalWrite(MOTOR_PIN, HIGH);
  digitalWrite(BUZZER_PIN, HIGH);
  delay(600); // Longer final pulse
  
  // Turn everything off
  digitalWrite(MOTOR_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
}

void playWarningAlert() {
  // A softer, quicker warning pulse for degrading posture
  
  // Single quick pulse
  digitalWrite(MOTOR_PIN, HIGH);
  // Optional: Leave buzzer off for warning, or do a very brief chirp
  // digitalWrite(BUZZER_PIN, HIGH); 
  delay(200);
  
  // Turn everything off
  digitalWrite(MOTOR_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
}
