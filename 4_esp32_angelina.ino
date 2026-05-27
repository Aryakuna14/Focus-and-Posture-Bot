// --- PIN CONFIGURATION ---
#define MOTOR_PIN 23    // Transistor Base
#define BUZZER_PIN 22   // Buzzer (+)

void setup() {
    Serial.begin(115200);
    // Force motor LOW before pinMode to prevent brief HIGH glitch on boot
    digitalWrite(MOTOR_PIN, LOW);
    pinMode(MOTOR_PIN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    
    // Test beep on startup
    tone(BUZZER_PIN, 1000, 200);
    Serial.println("Angelina Wired Node Ready.");
}

void alert(bool vibe) {
    if(vibe) digitalWrite(MOTOR_PIN, HIGH);
    tone(BUZZER_PIN, 1500, 300); // 1.5kHz beep
    delay(300);
    digitalWrite(MOTOR_PIN, LOW);
}


void loop() {
    if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();

        if (cmd == "TRIGGER") {
            alert(true);  // Vibe + Beep
        }
        else if (cmd == "BUZZ") {
            alert(false); // Just beep
        }
    }
}